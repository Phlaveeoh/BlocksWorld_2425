from PIL import Image, ImageDraw, ImageFont
# pip install pillow <- Comando che dovete fare se non avete Pillow :)

# Parametri finestra
width    = 1000
height   = 1000
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
block_font_color        = "#000"

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

roboticarm_horizontal_speed = 15


class RoboticArm:
    def __init__(self):
        self.posY = roboticarm_default_height
        self.posX = width/2
        
    def draw(self, frame):
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


    def slide_to(self, x):
        direction = 0
        if (x == self.posX):
            return
        elif (x > self.posX):
            direction += roboticarm_horizontal_speed
        else:
            direction -= roboticarm_horizontal_speed

        while abs(x - self.posX) > roboticarm_horizontal_speed:
            self.posX += direction
            draw_everything()

        self.posX = x
        draw_everything()


class Block:
    # PosX e PosY (quelli passati come parametri del costruttore, non quelli del self) non indicano la posizione nella gif ma nella matrice, occhio
    def __init__(self, value, posY, posX):
        self.value = value
        self.matrixPosX = posX
        self.matrixPosY = posY
        self.posX = (posX * (width // len(matrice_test[0]))) + block_left_offset
        self.posY = height - ((len(matrice_test) - posY) * block_height) - block_bottom_offset
        print(f"{value}: {self.posX}-{self.posY}")

    def draw(self, frame):
        draw = ImageDraw.Draw(frame)
        draw.rectangle([(self.posX, self.posY), (self.posX + block_width, self.posY + block_height)], fill=block_fill_color, outline=block_outline_color, width=block_outline_thickness)
        font = ImageFont.truetype("arial.ttf", block_font_size)
        
        # Dimensioni testo TextBBox
        text_bbox = draw.textbbox((0, 0), f"{self.value}", font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = self.posX + (block_width - text_width) / 2
        text_y = self.posY + (block_height - text_height - block_font_size/2) / 2
        
        draw.text((text_x, text_y), f"{self.value}", font=font, fill=block_font_color)


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

def create(matrix):
    for x in range(len(matrix)):
        for y in range(len(matrix[x])):
            value = matrix[y][x]
            
            if (value == 0):
                continue
            
            block = Block(value, x, y)
            blocks.append(block)
    
    draw_everything()

    # TEST MOVIMENTI DEL ROBOTTINO BELLISSIMO

    # Muove la mano robotica fino al 1° blocco che trova nel vettore
    robotic_arm.slide_to(blocks[0].posX + roboticarm_width/2)
    # Crea 20 frame senza cambiamenti
    for _ in range(20):
        draw_everything()

    # Fa il movimento completo fino a sinistra, poi fino a destra poi ritorna al centro e ricomincia la GIF
    robotic_arm.slide_to(0)
    robotic_arm.slide_to(width - block_left_offset - roboticarm_width)
    robotic_arm.slide_to(width/2)

    frames[0].save("BlocksWorld_Solution.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)

matrice_test = [[1,2,3,1,4,5],[0,0,0,0,0,7],[0,0,0,0,0,6],[0,0,0,0,0,0],[0,0,0,0,3,2],[0,0,0,0,8,9]]
create(matrice_test)