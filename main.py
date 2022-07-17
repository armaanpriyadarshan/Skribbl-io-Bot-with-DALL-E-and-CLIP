# Main script for the Skribbl.io bot
# Last edited on 7/13/2022
# Quote of the day: "U bum" - LeBron James

import itertools
import time
import cv2
import keyboard
import numpy as np
import mss
import mss.tools
import clip
import torch
import pyautogui
import matplotlib.pyplot as plt
from PIL import Image
from min_dalle import MinDalle
from pynput import keyboard
from queue import Queue

# PyAutoGUI's PAUSE is the number of seconds to pause after each PyAutoGUI function
# PyAutoGUI is used in this project to draw images on the Skribbl.io board, so clicking has to be fast to draw a 128x128 image in the short time frame
# Anything above 0 for the PAUSE made the drawing functionality too slow
pyautogui.PAUSE = 0

# A constant containing the bounding box in pixel coordinates of Skribbl.io elements, so they can be screenshotted
# Has coordinates for the monitor (which isn't used lol), the drawing board, and the blank word clue at the top
COORDS = {
        "monitor": [0, 0, 1920, 1080],
        "board": [256, 473, 835, 629],
        "word": [159, 930, 250, 46]
        # "word choices": [521, 667, 448, 58]
        # Originally wanted to use Tesseract to scan word choices presented when drawing is started
        # It didn't work, but I left coordinates for the word choices in there anyway
    }

# Loads the OpenAI CLIP model based on the ViT-B/32 transformer architecture
CLIP_MODEL, PREPROCESS = clip.load("ViT-B/32", device="cuda")
LABELS = {}

# The LABELS dictionary is filled with the possible words in Skribbl
# What happens in the program is that the CLIP model assigns a value to each label and ranks them to guess the word shown in the drawing
for line in open("_tokenization.txt").read().splitlines():
    LABELS[line] = "Pixel drawing of " + line

# Loads the pretrained DALL-E mini model (it doesn't generate the best images, but it's fast and runs on the 6 GB of VRAM my graphics card has)
# Change is_mega to True to use the DALL-E mega model
# Hopefully, DALLE-2 is going to go public soon which would allow significant room for improvement
DALLE_MODEL = MinDalle(
    models_root='./pretrained',
    dtype=torch.float16,
    is_mega=False,
    is_reusable=True,
)

# RGB value of each Skribbl color mapped to a tuple with pixel coordinates of the color on the screen
# Used for dithering the generated images and also for coloring them in Skribbl.io
PALETTE = {
    (255, 255, 255): (585, 861),
    (190, 194, 189): (611, 861),
    (239, 21, 10): (633, 861),
    (252, 111, 3): (656, 861),
    (254, 229, 0): (680, 861),
    (2, 205, 3): (704, 861),
    (1, 179, 254): (731, 861),
    (38, 33, 208): (754, 861),
    (164, 1, 184): (777, 861),
    (212, 125, 173): (802, 861),
    (158, 83, 42): (825, 861),
    (1, 0, 1): (585, 885),
    (77, 76, 77): (611, 885),
    (116, 10, 6): (633, 885),
    (197, 56, 0): (656, 885),
    (231, 163, 1): (680, 885),
    (2, 85, 17): (704, 885),
    (1, 85, 159): (731, 885),
    (13, 6, 101): (754, 885),
    (84, 1, 104): (777, 885),
    (164, 85, 117): (802, 885),
    (99, 49, 13): (825, 885)
}

# Global variable used in the keybind logic
running = False


def screenshot(coord_list):
    """ Screenshots a given coord_list (a pixel coordinate bounding box) with the MSS library and returns the screenshot for further processing """

    sct = mss.mss()
    bb = {"top": coord_list[0], "left": coord_list[1], "width": coord_list[2], "height": coord_list[3]}

    output = sct.grab(bb)

    return output


def get_blanks(image):
    """ Given an image of the blank word (look at test.png or test2.png in tests), returns the number of underscores in the image """

    # Only works with blank words that don't have letters clued in
    # Initially wanted to do this with Tesseract, but it didn't work, so I used OpenCV's LineSegmentDetector

    # Converts the image to a numpy array
    image = np.array(image)

    # Image preprocessing (greyscale, thresholding, and upscaling)
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(greyscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Upscales the image by 500 percent to make the underscores bigger and easier to detect
    # Haven't really tested the minimum upscale size needed to remain accurate but 500 works so
    scale_percent = 500  # percent of original size
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)

    # Thresholds the image again with binary inversion to make a bitonal image and improve dilation + LineSegmentDetector accuracy
    thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Dilates the image which is basically just making the underscores thinner and more distinct
    kernel = np.ones((10, 10), np.uint8)
    dilated_image = cv2.dilate(thresh, kernel)

    # Detection of the underscores/line segments
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(dilated_image)[0]

    # cv2.imshow("Test", lsd.drawSegments(dilated_image, lines))
    # cv2.waitKey()
    # Returns the length of lines/2 because that's how it works apparently
    return int(len(list(lines)) / 2)


def clip_rank(model, preprocess, image, labels, word_length):
    """ Given an image a Skribbl.io drawing and the length of the word it's supposed to depict, returns a sorted list of the probabilities that each possible word matches the image """

    # Starts by narrowing down the list of possible words to those that are the same length as the blank word clue
    labels_subset = {key: value for key, value in labels.items() if len(key.strip()) == word_length}

    # Image processing required to digest the screenshot returned by the screenshot() function
    image = Image.frombytes("RGB", image.size, image.bgra, "raw", "BGRX")
    processed_image = preprocess(image.convert("RGB")).unsqueeze(0).cuda()

    # Tokenization of the label subset
    text = clip.tokenize([label for label in labels_subset.values()]).cuda()

    # Referenced from RoboFlow's CLIP Colab notebook
    # CLIP ranking
    with torch.no_grad():
        logits_per_image, logits_per_text = model(processed_image, text)
        # Sorts the list of labels by probability such that the first label matches the drawing the best
        probs = sorted({key: value for key, value in
                        zip(labels_subset.keys(), list(logits_per_image.softmax(dim=-1).cpu().numpy())[0])}.items(),
                       key=lambda x: x[1], reverse=True)
    # Returns the list of probabilities
    return probs


def visualize_guesses(probs):
    """ Visualizes the top 5 guesses returned by the clip_rank function with a Matplotlib plot """

    top_five = dict(probs[:5])

    # A horizontal bar chart with the label (for example: sunglasses, headphones, etc.) on the y-axis and the confidence from 0-100 on the x-axis
    plt.figure(figsize=(8, 4.8))
    plt.barh(list(top_five.keys()), [100 * value for value in top_five.values()])

    plt.ylabel("Guesses")
    plt.xlabel("Confidence")
    plt.gca().invert_yaxis()

    plt.show()


def generate_image(model, prompt):
    """ Generates and returns an image with DALL-E given a prompt """

    image = model.generate_image(
        text=prompt,
        seed=-1,
        grid_size=1,
        log2_k=6,
        log2_supercondition_factor=5,
        is_verbose=False
    )

    return image


def dither_and_draw(image, palette):
    """ Given an image from the generate_image function, dithers it (reduces the color depth to the palette in Skribbl.io) and draws it on the Skribbl.io drawing board using PyAutoGUI """

    # Resizes the 256x256 image from DALL-E to 128x128 to reduce the number of pixels in it and speed up drawing
    resized_image = image.resize((round(0.5 * image.size[0]), round(0.5 * image.size[1])))

    # A palette image for PIL because that's how dithering with quantization works
    palette_image = Image.new('P', (484, 484))
    palette_image.putpalette(list(itertools.chain(*palette.keys())))

    # Quantization/dithering of the image
    image_dithered = resized_image.quantize(palette=palette_image, dither=1).convert("RGB")
    # image_dithered.show()

    # Start coordinates for drawing on the board
    # Used to map where each pixel in the drawing goes on the screen
    x_start, y_start = (700, 300)

    # Gets dimensions and pixel data from the dithered image
    width, height = image_dithered.size
    pixels = image_dithered.load()

    # Gets color data from the dither image with .getcolors()
    # This returns the number of pixels corresponding with each color in the image which is used to make a sorted list of colors by frequency which goes from most frequent to the least frequent
    image_colors = {v: k for k, v in dict(image_dithered.getcolors()).items()}
    colors_by_frequency = [i[0] for i in sorted(image_colors.items(), key=lambda x: x[1], reverse=True)]

    # Drawing logic to draw the image hastily with little pixel overlap
    # The drawing also seems much bigger on the board than the 128x128 dithered image
    # Goes color by color from most frequent color to the least frequent (so you don't have to switch color every pixel)
    for color in colors_by_frequency:
        # Skips the color if it's white
        if color == (255, 255, 255):
            continue
        # Switches color by clicking the coordinates in the palette dictionary
        pyautogui.click(palette[color][0], palette[color][1])
        # Iterates through each pixel in every row and draws them if they are the right color
        for y in range(3, height * 3, 3):
            for x in range(3, width * 3, 3):
                if pixels[int(x / 3), int(y / 3)] == color:
                    pyautogui.click(x_start + x, y_start + y)

        # I found that the bot draws faster than what Skribbl.io can keep up with, so a sleep is needed to maintain the quality of the drawing
        # The sleep adjusts accordingly for the number of pixels encompassed by the color in the iteration, so colors with more pixels give Skribbl enough time to catch up with mouse clicks
        time.sleep(image_colors[color]/1000)


def on_press(key, queue):
    """ Pynput listener function for control flow of the program in correspondence to key bindings """

    # Pushes different events to a queue to be handled by the main thread
    # Keybindings go "S" to get the length of a blank word, "G" to visualize guesses for the word, and "D" to generate an image and draw it given an input word
    # Escape is used to push a "stop" event to stop the program and Insert is used to toggle typing, so you can type without the bot interpreting key presses as input
    # Global running variable to stop the listener from pushing events when needed
    global running
    try:
        if key.char == "s" and not running:
            queue.put("blanks")

        if key.char == "g" and not running:
            queue.put("guess")

        if key.char == "d" and not running:
            queue.put("draw")

    except AttributeError:
        if key == keyboard.Key.insert:
            running = not running
            print("Typing toggled") if running else print("Typing untoggled")

        if key == keyboard.Key.esc:
            queue.put("stop")
            return False


def main():
    torch.cuda.empty_cache()
    event_queue = Queue()

    # Welcome message that prints the keybindings
    print("\nReady!\n- If you are guessing, press S to get the number of letters and then hit G afterwards to get the word\n- If you are drawing, hit D\n- Press Insert to toggle typing so the program doesn't try to track key presses when you want to type in Skribbl")

    # Starts the keyboard listener and provides the event_queue as the output queue
    listener = keyboard.Listener(on_press=lambda key: on_press(key, event_queue))
    listener.start()

    # Global running variable again to stop the listener from reading key presses in the middle of an operation
    # num_letters variable for reassignment to the number of letters in the blank word
    global running
    num_letters = 0

    while True:
        # Reads the event from the event_queue and has corresponding functionality for each event
        event = event_queue.get()
        if event == "blanks":
            running = True
            print('\nGetting the number of letters in the word...')
            num_letters = get_blanks(screenshot(COORDS["word"]))
            print("There are " + str(num_letters) + " letters")

        if event == "guess":
            running = True
            print("\nGuessing word with word length " + str(num_letters))
            probs = clip_rank(CLIP_MODEL, PREPROCESS, screenshot(COORDS["board"]), LABELS, num_letters)
            print("Visualizing top 5 guesses...")
            visualize_guesses(probs)

        if event == "draw":
            running = True
            prompt = input("\nWord: ")
            print(f"Generating a picture of {prompt}...")
            image = generate_image(DALLE_MODEL, prompt)
            print("Drawing the image...")
            dither_and_draw(image, PALETTE)

        # Breaks the while loop when the "stop" event is read
        if event == "stop":
            break

        running = False


if __name__ == '__main__':
    main()
