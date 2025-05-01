from PIL import Image, ImageDraw, ImageFont
import random
import time

# Parametri finestra
width    = 650
height   = 650
bg_color = (255, 255, 255)
background_image_path = ".\\src\\Background_01.png"

# Parametri blocchi
block_height            = 75
block_width             = 75
block_bottom_offset     = 10
block_left_offset       = 10
block_sprites_path       = [
    ".\\src\\Block_Sprite_01.png", # Sprite 01
    ".\\src\\Block_Sprite_02.png", # Sprite 02
    ".\\src\\Block_Sprite_03.png"  # Sprite 03
]
block_number_sprites_path = [
    ".\\src\\Number_Sprite_01.png", # Sprite 01
    ".\\src\\Number_Sprite_02.png", # Sprite 02
    ".\\src\\Number_Sprite_03.png", # Sprite 03
    ".\\src\\Number_Sprite_04.png", # Sprite 04
    ".\\src\\Number_Sprite_05.png", # Sprite 05
    ".\\src\\Number_Sprite_06.png", # Sprite 06
]
block_grid_rows = 6
block_grid_columns = 6

# Parametri binario mano robotica
roboticarm_track_height    = 50
roboticarm_track_thickness = 6
roboticarm_track_color     = "#ffa900"

# Parametri gancio mano robotica
roboticarm_slider_width  = 50
roboticarm_slider_height = 30
roboticarm_slider_color  = "#A0A0A0"

# Parametri mano robotica
roboticarm_default_height = 125
roboticarm_color          = "#A0A0A0"
roboticarm_thickness      = 7
roboticarm_width          = block_width + roboticarm_track_thickness + 2

# Parametri 'dita' mano robotica
roboticarm_claw_height    = 30
roboticarm_claw_thickness = 5

# Parametri braccio della mano robotica
roboticarm_arm_thickness = 4
roboticarm_arm_color     = "#505050"

# Velocità movimenti
roboticarm_horizontal_speed = 40
roboticarm_vertical_speed = 40

gif_path = ".\\static\\result\\BlocksWorld_Solution.gif"
frame_duration = 30

# Ottimizzazione del codice e del caricamento dei file
SPRITE_CACHE = {}
class RoboticArm:
    def __init__(self):
        self.posY = roboticarm_default_height
        self.posX = width/2
        self.grabbed_block = None
        
    def draw(self, frame):
        half_track_thickness = roboticarm_track_thickness/2
        half_slider_width = roboticarm_slider_width/2
        half_slider_height = roboticarm_slider_height/2
        half_arm_width = roboticarm_width/2
        half_arm_thickness = roboticarm_thickness
        half_armstructure_thickness = roboticarm_arm_thickness/2 # E' il braccio del braccio robotico, non sapevo come altro chiamarlo :)

        draw = ImageDraw.Draw(frame)
        self.move_grabbed_block()
        # Binario
        draw.rectangle([(0, roboticarm_track_height - half_track_thickness), (width, roboticarm_track_height + half_track_thickness)], fill=roboticarm_track_color)

        # Mano robotica
        draw.rectangle([(self.posX - half_arm_width, self.posY - half_arm_thickness), (self.posX + half_arm_width, self.posY + half_arm_thickness)], fill=roboticarm_color)
        # Gancio sinistro
        draw.rectangle([(self.posX - half_arm_width, self.posY + half_arm_thickness), (self.posX - half_arm_width + roboticarm_claw_thickness, self.posY + half_arm_thickness + roboticarm_claw_height)], fill=roboticarm_color)
        # Gancio Destro
        draw.rectangle([(self.posX + half_arm_width - roboticarm_claw_thickness, self.posY + half_arm_thickness), (self.posX + half_arm_width, self.posY + half_arm_thickness + roboticarm_claw_height)], fill=roboticarm_color)
        # Braccio mano robotica
        draw.rectangle([(self.posX - half_armstructure_thickness, roboticarm_track_height+half_track_thickness), (self.posX + half_armstructure_thickness, self.posY - half_arm_thickness)], fill=roboticarm_arm_color)

        draw.rectangle([
            (self.posX - half_slider_width, roboticarm_track_height + half_track_thickness - half_slider_height),
            (self.posX + half_slider_width, roboticarm_track_height + half_track_thickness + half_slider_height)    
        ], fill=roboticarm_slider_color)


    def slide_horizzontally_to(self, x):
        direction = (1 if x > self.posX else -1) * roboticarm_horizontal_speed
        while abs(x - self.posX) > roboticarm_horizontal_speed:
            self.posX += direction
            if abs(x - self.posX) <= roboticarm_horizontal_speed:
                self.posX = x
            self.move_grabbed_block()
            draw_everything()
        gif_sleep(100)

    def slide_vertically_to(self, y):
        direction = (1 if y > self.posY else -1) * roboticarm_vertical_speed
        while abs(y - self.posY) > roboticarm_vertical_speed:
            self.posY += direction
            if abs(y - self.posY) <= roboticarm_vertical_speed:
                self.posY = y
            self.move_grabbed_block()
            draw_everything()
        gif_sleep(100)

    def move_grabbed_block(self):
        if (self.grabbed_block == None):
            return
        self.grabbed_block.posY = self.posY
        self.grabbed_block.posX = int(self.posX - roboticarm_width/2 + roboticarm_claw_thickness)

    def grab(self, block):
        self.slide_horizzontally_to(block.posX + roboticarm_width/2 - roboticarm_claw_thickness/2)
        self.slide_vertically_to(block.posY)
        self.grabbed_block = block
        self.slide_vertically_to(roboticarm_default_height)

    def release(self, matrixX, matrixY):
        frameX = convert_matrixX_into_frameX(matrixX) + roboticarm_width/2 - roboticarm_claw_thickness
        frameY = convert_matrixY_into_frameY(matrixY)

        self.slide_horizzontally_to(frameX)
        self.slide_vertically_to(frameY)

        self.grabbed_block.matrixPosX = matrixX
        self.grabbed_block.matrixPosY = matrixY
        self.grabbed_block = None

        self.slide_vertically_to(roboticarm_default_height)


class Block:
    # PosX e PosY (quelli passati come parametri del costruttore, non quelli del self) non indicano la posizione nella gif ma nella matrice, occhio
    def __init__(self, value, posY, posX):
        self.value = value
        self.blockSprite = random.choice(block_sprites_path)
        self.numberSprite = block_number_sprites_path[value-1]
        self.matrixPosX = posX
        self.matrixPosY = posY
        self.posX = convert_matrixX_into_frameX(posX)
        self.posY = convert_matrixY_into_frameY(posY)

    def draw(self, frame):
        if self.blockSprite not in SPRITE_CACHE:
            SPRITE_CACHE[self.blockSprite] = Image.open(self.blockSprite).convert("RGBA")
        block_img = SPRITE_CACHE[self.blockSprite]
        frame.paste(block_img, (self.posX, self.posY, self.posX + block_width, self.posY + block_height), block_img)
    
        # Carica in cache lo sprite del numero, se non già presente
        if self.numberSprite not in SPRITE_CACHE:
            SPRITE_CACHE[self.numberSprite] = Image.open(self.numberSprite).convert("RGBA")
        number_img = SPRITE_CACHE[self.numberSprite]
    
        # Ottieni dimensioni dell'immagine del numero
        number_w, number_h = number_img.size
        # Calcola le coordinate per centrare l'immagine all'interno del blocco
        number_posX = self.posX + (block_width - number_w) // 2
        number_posY = self.posY + (block_height - number_h) // 2
    
        # Incolla l'immagine del numero al centro del blocco
        frame.paste(number_img, (number_posX, number_posY, number_posX + number_w, number_posY + number_h), number_img)


    def has_this_matrix_coordinates(self, x, y):
        return self.matrixPosY == y and self.matrixPosX == x


# Lista dei frame
frames = []
# Lista blocchi
blocks = []
# Mano robotica
robotic_arm = RoboticArm()

background_image = Image.open(background_image_path).convert("RGB").resize((width, height))
def draw_everything():
    frame = Image.new("RGB", (width, height), bg_color)
    #frame = background_image.copy()
    for block in blocks:
        block.draw(frame)
    robotic_arm.draw(frame)
    frames.append(frame)

def gif_sleep(pause_duration):
    for _ in range(pause_duration // frame_duration):
        frames.append(frames[-1])

def create(matrix, moves):
    start_time = time.time()
    
    for x in range(len(matrix)):
        for y in range(len(matrix[x])):
            value = matrix[y][x]
            
            if (value == 0):
                continue
            
            block = Block(value, x, y)
            blocks.append(block)
    
    draw_everything()

    for move in moves:
        x, y, newX, newY = move
        block = get_block_from_matrix_coordinates(x, y)
        robotic_arm.grab(block)
        robotic_arm.release(newX, newY)

    robotic_arm.slide_horizzontally_to(width/2)

    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=frame_duration, loop=0, optimize=True)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"La GIF è stata generata in {execution_time:.2f} secondi")

    reset()

    return gif_path

def get_block_from_matrix_coordinates(x, y):
    for block in blocks:
        if block.has_this_matrix_coordinates(x, y):
            return block
    return None

def convert_matrixX_into_frameX(x):
    return (x * (width // block_grid_columns)) + block_left_offset

def convert_matrixY_into_frameY(y):
    return height - ((block_grid_rows - y) * block_height) - block_bottom_offset

def reset():
    frames = []
    blocks = []
    robotic_arm = RoboticArm()