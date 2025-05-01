from PIL import Image, ImageDraw, ImageFont
import random
# pip install pillow <- Comando che dovete fare se non avete Pillow :)

# Parametri finestra
width    = 1000
height   = 800
bg_color = (255, 255, 255)

# Parametri blocchi
block_height            = 100
block_width             = 100
block_bottom_offset     = 10
block_left_offset       = 10
block_outline_color     = "#000"
block_outline_thickness = 5
block_fill_color        = "#fff"
block_font_size         = 30
block_font_color        = "#fff"
block_sprites_path       = [
    ".\\src\\Block_Sprite_01.png", # Sprite 01
    ".\\src\\Block_Sprite_02.png", # Sprite 02
    ".\\src\\Block_Sprite_03.png"  # Sprite 03
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
roboticarm_horizontal_speed = 20
roboticarm_vertical_speed = 20

git_path = ".\\static\\result\\BlocksWorld_Solution.gif"


class RoboticArm:
    def __init__(self):
        self.posY = roboticarm_default_height
        self.posX = width/2
        self.grabbed_block = None
        
    def draw(self, frame):
        # Aggiorna la posizione del blocco afferrato
        self.move_grabbed_block()
        
        half_track_thickness = roboticarm_track_thickness/2
        half_slider_width = roboticarm_slider_width/2
        half_slider_height = roboticarm_slider_height/2
        half_arm_width = roboticarm_width/2
        half_arm_thickness = roboticarm_thickness
        half_armstructure_thickness = roboticarm_arm_thickness/2 # E' il braccio del braccio robotico, non sapevo come altro chiamarlo :)

        draw = ImageDraw.Draw(frame)
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

        # Carrello che tiene la mano robotica | Per ora è centrato secondo la dimensione dello schermo, ma è errato.
        # Deve essere la mano robotica quella centrata, il resto deve adattarsi in base a lei.
        draw.rectangle([
            (self.posX - half_slider_width, roboticarm_track_height + half_track_thickness - half_slider_height),
            (self.posX + half_slider_width, roboticarm_track_height + half_track_thickness + half_slider_height)    
        ], fill=roboticarm_slider_color)


    def slide_horizzontally_to(self, x):
        direction = (1 if x > self.posX else -1) * roboticarm_horizontal_speed

        while abs(x - self.posX) > roboticarm_horizontal_speed:
            self.posX += direction
            self.move_grabbed_block()
            draw_everything()

        self.posX = x
        draw_everything()
        gif_sleep(10)

    def slide_vertically_to(self, y):
        direction = (1 if y > self.posY else -1) * roboticarm_vertical_speed
        while abs(y - self.posY) > roboticarm_vertical_speed:
            self.posY += direction
            self.move_grabbed_block()
            draw_everything()

        self.posY = y
        draw_everything()
        gif_sleep(10)

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
        self.sprite = random.choice(block_sprites_path)
        self.matrixPosX = posX
        self.matrixPosY = posY
        self.posX = convert_matrixX_into_frameX(posX)
        self.posY = convert_matrixY_into_frameY(posY)
        print(f"{value}: {self.posX}-{self.posY}")

    def draw(self, frame):
        draw = ImageDraw.Draw(frame)
        image = Image.open(self.sprite)

        frame.paste(image, (self.posX, self.posY, self.posX + block_width, self.posY + block_height), image)
        #draw.rectangle([(self.posX, self.posY), (self.posX + block_width, self.posY + block_height)], fill=block_fill_color, outline=block_outline_color, width=block_outline_thickness)
        
        font = ImageFont.truetype("arial.ttf", block_font_size)
        
        # Dimensioni testo TextBBox
        text_bbox = draw.textbbox((0, 0), f"{self.value}", font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = self.posX + (block_width - text_width) / 2
        text_y = self.posY + (block_height - text_height - block_font_size/2) / 2
        
        draw.text((text_x, text_y), f"{self.value}", font=font, fill=block_font_color)

    def has_this_matrix_coordinates(self, x, y):
        return self.matrixPosY == y and self.matrixPosX == x


# Lista dei frame
frames = []
# Lista blocchi
blocks = []
# Mano robotica
robotic_arm = RoboticArm()

def draw_everything():
    frame = Image.new("RGB", (width, height), bg_color)
    for i in range(len(blocks)):
        blocks[i].draw(frame)
    robotic_arm.draw(frame)
    frames.append(frame)

def gif_sleep(frames_length):
    for i in range(frames_length):
        draw_everything()

def create(matrix, moves):
    print(moves)
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
        print(block)


    frames[0].save(git_path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    return git_path

def get_block_from_matrix_coordinates(x, y):
    for block in blocks:
        if block.has_this_matrix_coordinates(x, y):
            return block
    return None

def convert_matrixX_into_frameX(x):
    return (x * (width // block_grid_columns)) + block_left_offset

def convert_matrixY_into_frameY(y):
    return height - ((block_grid_rows - y) * block_height) - block_bottom_offset