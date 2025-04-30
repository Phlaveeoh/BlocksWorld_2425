from PIL import Image, ImageDraw, ImageFont

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

# Lista dei frame
frames = []





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




class RoboticArm:
    def draw(self, frame):
        half_frame_width = width/2
        half_track_thickness = roboticarm_track_thickness/2
        half_slider_width = roboticarm_slider_width/2
        half_slider_height = roboticarm_slider_height/2

        draw = ImageDraw.Draw(frame)
        # Binario
        draw.rectangle([(0, roboticarm_track_height - half_track_thickness), (width, roboticarm_track_height + half_track_thickness)], fill=roboticarm_track_color)
        # Carrello che tiene la mano robotica | Per ora è centrato secondo la dimensione dello schermo, ma è errato.
        # Deve essere la mano robotica quella centrata, il resto deve adattarsi in base a lei.
        draw.rectangle([
            (half_frame_width - half_slider_width, roboticarm_track_height + half_track_thickness - half_slider_height),
            (half_frame_width + half_slider_width, roboticarm_track_height + half_track_thickness + half_slider_height)    
        ], fill=roboticarm_slider_color)




def create(matrix):
    blocks = []
    frame = Image.new("RGB", (width, height), bg_color)
    robotic_arm = RoboticArm()

    robotic_arm.draw(frame)

    for x in range(len(matrix)):
        for y in range(len(matrix[x])):
            value = matrix[y][x]
            
            if (value == 0):
                continue
            
            block = Block(value, x, y)
            blocks.append(block)
            block.draw(frame)
    
    frames.append(frame)
    frames[0].save("BlocksWorld_Solution.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)

matrice_test = [[1,2,3,1,4,5],[0,0,0,0,0,7],[0,0,0,0,0,6],[0,0,0,0,0,0],[0,0,0,0,3,2],[0,0,0,0,8,9]]
create(matrice_test)