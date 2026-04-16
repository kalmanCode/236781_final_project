class ImageObject:
    def __init__(self, name, image, prompts, center_x=256, center_y=256):
        self.name = name
        self.image = image
        self.prompts = prompts
        self.center_x = center_x
        self.center_y = center_y

class MaskObject:
    def __init__(self, name, image):
        self.name = name
        self.image = image
