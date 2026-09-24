# WITH VOICE ASSISTANT + LIVE CAMERA IMAGE CAPTURE + SHELF LIFE INTEGRATION
import os
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pyttsx3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === CONFIG ===
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 3
MODEL_PATH = "mobilenet_food_model.h5"
VIDEO_URL = 'http://192.168.1.72:4747/video'  # Replace with actual IP Webcam link

# === Paths ===
train_dir = "D:\YOLOV5 data\random data.txt"
val_dir = "D:\YOLOV5 data\random data.txt"

# === Load Class Labels ===
class_labels = sorted(os.listdir(train_dir))

# === Data Generators ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === Model Handling ===
if not os.path.exists(MODEL_PATH):
    base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(len(class_labels), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    model.save(MODEL_PATH)
else:
    model = load_model(MODEL_PATH)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# === Audio Engine ===
speaker = pyttsx3.init()
speaker.setProperty('rate', 150)

def speak(text):
    speaker.say(text)
    speaker.runAndWait()

# === Recipes and Shelf Life ===
recipes = {
    "tomato":{
        "Low": ["Tomato Soup", "Tomato Chutney"],
        "Medium": ["Pasta Sauce", "Tomato Rice"],
        "High": ["Fresh Salad", "Tomato Bruschetta"]
    },
    "milk": {
        "Low": ["Paneer", "Milkshake"],
        "Medium": ["Pudding", "Custard"],
        "High": ["Tea", "Coffee"]
    },
    "banana": {
        "Low": ["Banana Pancakes", "Banana Ice Cream"],
        "Medium": ["Banana Muffins", "Banana Bread"],
        "High": ["Smoothie", "Fruit Salad"]
    },
    "bellpepper": {
        "Low": ["Stuffed Bell Peppers", "Bell Pepper Soup"],
        "Medium": ["Bell Pepper Stir Fry", "Bell Pepper Pasta"],
        "High": ["Bell Pepper Salad", "Grilled Bell Pepper Skewers"]
        },
    "cabbage": {
        "Low": ["Cabbage Soup", "Cabbage Paratha"],
        "Medium": ["Stir-fried Cabbage", "Cabbage Curry"],
        "High": ["Coleslaw", "Cabbage Salad"]
    },
    "capsicum": {
        "Low": ["Capsicum Chutney", "Stuffed Capsicum"],
        "Medium": ["Capsicum Fried Rice", "Capsicum Masala"],
        "High": ["Capsicum Sandwich", "Capsicum Salad"]
    },
    "carrot": {
        "Low": ["Carrot Halwa", "Carrot Soup"],
        "Medium": ["Carrot Stir Fry", "Carrot Rice"],
        "High": ["Carrot Salad", "Carrot Sticks with Hummus"]
    },
    "cauliflower": {
        "Low": ["Cauliflower Soup", "Cauliflower Mash"],
        "Medium": ["Gobi Manchurian", "Cauliflower Curry"],
        "High": ["Roasted Cauliflower Bites", "Cauliflower Stir Fry"]
    },
    "chilli pepper": {
        "Low": ["Green Chili Pickle", "Stuffed Chilies"],
        "Medium": ["Chili Pakora", "Chili Garlic Sauce"],
        "High": ["Spicy Salsa", "Chili Topping for Nachos"]
    },
    "corn": {
    "Low": ["Corn Soup", "Corn Cutlet"],
    "Medium": ["Corn Fried Rice", "Corn Curry"],
        "High": ["Boiled Corn", "Grilled Corn on the Cob"]
    },
    "cucumber": {
        "Low": ["Cucumber Curry", "Cucumber Stew"],
        "Medium": ["Cucumber Raita", "Cucumber Sandwich"],
        "High": ["Cucumber Salad", "Cucumber Juice"]
    },
    "eggplant": {
        "Low": ["Eggplant Puree", "Baingan Bharta"],
        "Medium": ["Stuffed Eggplant", "Eggplant Curry"],
        "High": ["Grilled Eggplant", "Eggplant Stir Fry"]
    },
    "garlic": {
        "Low": ["Garlic Paste", "Garlic Pickle"],
        "Medium": ["Garlic Butter Sauce", "Roasted Garlic Dip"],
        "High": ["Garlic Bread", "Garlic Tadka for Dals"]
    },
    "ginger": {
        "Low": ["Ginger Pickle", "Dry Ginger Powder"],
        "Medium": ["Ginger Tea", "Ginger Syrup"],
        "High": ["Ginger Juice", "Ginger-Lemon Water"]
    },
    "jalepeno": {
    "Low": ["Stuffed Jalapeños", "Jalapeño Chutney"],
    "Medium": ["Pickled Jalapeños", "Jalapeño Cornbread"],
    "High": ["Jalapeño Topping", "Jalapeño Dip"]
    },
    "onion": {
        "Low": ["Onion Soup", "Caramelized Onion Paste"],
        "Medium": ["Onion Pakora", "Onion Curry"],
        "High": ["Onion Salad", "Onion Sandwich"]
    },
    "paprika": {
        "Low": ["Paprika Sauce", "Paprika Marinade"],
        "Medium": ["Paprika Potato Roast", "Paprika Pasta"],
        "High": ["Paprika Popcorn", "Sprinkled on Salads"]
    },
    "peas": {
        "Low": ["Peas Soup", "Peas Cutlet"],
        "Medium": ["Matar Paneer", "Peas Pulao"],
        "High": ["Peas Salad", "Boiled Peas Stir Fry"]
    },
    "potato": {
        "Low": ["Potato Soup", "Mashed Potatoes"],
        "Medium": ["Aloo Paratha", "Aloo Curry"],
        "High": ["French Fries", "Boiled Potato Salad"]
    },
    "raddish": {
        "Low": ["Radish Curry", "Radish Soup"],
        "Medium": ["Radish Stir Fry", "Radish Pickle"],
        "High": ["Radish Salad", "Radish Paratha"]
    },
    "soybeans": {
        "Low": ["Soybean Curry", "Soybean Cutlets"],
        "Medium": ["Soybean Stir Fry", "Soybean Rice"],
        "High": ["Boiled Soybeans", "Soybean Sprout Salad"]
    },
    "spinach": {
        "Low": ["Spinach Soup", "Spinach Puree"],
        "Medium": ["Palak Paneer", "Spinach Pasta"],
        "High": ["Spinach Salad", "Spinach Smoothie"]
    },
    "sweetcorn": {
        "Low": ["Creamed Corn", "Sweet Corn Soup"],
        "Medium": ["Sweet Corn Fried Rice", "Corn Chaat"],
        "High": ["Boiled Sweet Corn", "Corn Salad"]
    },
    "sweetpotato": {
        "Low": ["Sweet Potato Soup", "Sweet Potato Mash"],
        "Medium": ["Sweet Potato Curry", "Sweet Potato Fries"],
        "High": ["Baked Sweet Potato", "Sweet Potato Salad"]
    },
    "turnip": {
        "Low": ["Turnip Soup", "Turnip Mash"],
        "Medium": ["Turnip Curry", "Stir-fried Turnip"],
        "High": ["Boiled Turnip Salad", "Turnip Pickle"]
    },
    "grapes": {
    "Low": ["Grape Jam", "Grape Juice Concentrate"],
    "Medium": ["Grape Jelly", "Grape Compote"],
    "High": ["Fruit Salad", "Fresh Grapes with Cheese"]
},
"kiwi": {
    "Low": ["Kiwi Jam", "Kiwi Sauce"],
    "Medium": ["Kiwi Smoothie", "Kiwi Pudding"],
    "High": ["Fresh Kiwi Slices", "Kiwi Fruit Salad"]
},
"lemon": {
    "Low": ["Lemon Pickle", "Preserved Lemon"],
    "Medium": ["Lemon Syrup", "Lemon Tart"],
    "High": ["Lemon Water", "Lemon Tea"]
},
"lettuce": {
    "Low": ["Lettuce Soup", "Sautéed Lettuce"],
    "Medium": ["Lettuce Wraps", "Grilled Lettuce"],
    "High": ["Lettuce Salad", "Lettuce Sandwich"]
},
"mango": {
    "Low": ["Mango Pickle", "Mango Chutney"],
    "Medium": ["Mango Pulp", "Mango Custard"],
    "High": ["Fresh Mango Slices", "Mango Smoothie"]
},
"orange": {
    "Low": ["Orange Marmalade", "Candied Orange Peel"],
    "Medium": ["Orange Cake", "Orange Popsicles"],
    "High": ["Orange Juice", "Orange Segments in Salad"]
},
"pear": {
    "Low": ["Pear Jam", "Stewed Pears"],
    "Medium": ["Pear Tart", "Pear Muffins"],
    "High": ["Fresh Pear Slices", "Pear Salad"]
},
"pineapples": {
    "Low": ["Pineapple Jam", "Pineapple Chutney"],
    "Medium": ["Pineapple Upside-Down Cake", "Grilled Pineapple"],
    "High": ["Pineapple Juice", "Pineapple Salad"]
},
"pomegranate": {
    "Low": ["Pomegranate Syrup", "Pomegranate Molasses"],
    "Medium": ["Pomegranate Glaze", "Pomegranate Yogurt"],
    "High": ["Fresh Pomegranate Seeds", "Pomegranate Salad"]
},
"watermelon": {
    "Low": ["Watermelon Jelly", "Watermelon Ice Cream"],
    "Medium": ["Watermelon Smoothie", "Watermelon Sorbet"],
    "High": ["Watermelon Slices", "Watermelon Juice"]
}

}

# === GUI & Detection ===
detected_item = None
captured_image_path = "captured_from_live.jpg"

root = tk.Tk()
root.title("Smart Fridge - Interactive Detection")
root.geometry("600x500")
root.config(bg="#f0f8ff")

heading = tk.Label(root, text="Smart Fridge System", font=("Helvetica", 20, "bold"), bg="#f0f8ff", fg="#333")
heading.pack(pady=10)

frame = tk.Frame(root, bg="#e6f2ff", padx=10, pady=10, relief=tk.RIDGE, bd=2)
frame.pack(pady=10)

img_label = tk.Label(frame, text="Upload or use live camera", bg="#e6f2ff")
img_label.pack()

panel = tk.Label(frame)
panel.pack(pady=10)

temp_label = tk.Label(frame, text="Temperature (°C):", bg="#e6f2ff")
temp_label.pack()
temp_entry = tk.Entry(frame)
temp_entry.pack()

humidity_label = tk.Label(frame, text="Humidity (%):", bg="#e6f2ff")
humidity_label.pack()
humidity_entry = tk.Entry(frame)
humidity_entry.pack()

def upload_image():
    global detected_item
    file_path = filedialog.askopenfilename()
    if file_path:
        img = load_img(file_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]
        idx = np.argmax(prediction)
        label = class_labels[idx]
        confidence = prediction[idx] * 100
        detected_item = label

        img_display = Image.open(file_path)
        img_display = img_display.resize((150, 150))
        img_tk = ImageTk.PhotoImage(img_display)
        panel.config(image=img_tk)
        panel.image = img_tk

        speak(f"Detected item is {label}")
        messagebox.showinfo("Prediction", f"Detected: {label} ({confidence:.2f}%)")

def use_captured_image():
    global captured_image_path, detected_item
    if not os.path.exists(captured_image_path):
        speak("No image captured from live camera yet.")
        messagebox.showwarning("Warning", "No image captured from live detection.")
        return

    img = load_img(captured_image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    idx = np.argmax(prediction)
    label = class_labels[idx]
    detected_item = label
    confidence = prediction[idx] * 100

    img_display = Image.open(captured_image_path)
    img_display = img_display.resize((150, 150))
    img_tk = ImageTk.PhotoImage(img_display)
    panel.config(image=img_tk)
    panel.image = img_tk

    speak(f"Captured item is {label}")
    messagebox.showinfo("Captured Prediction", f"Item: {label} ({confidence:.2f}%)")

def predict_shelf_life():
    try:
        global detected_item
        temp = float(temp_entry.get())
        humidity = float(humidity_entry.get())

        if temp > 30 or humidity > 60:
            life = "Low"
            range_text = "1-2 days"
        elif 20 < temp <= 30 and 40 < humidity <= 60:
            life = "Medium"
            range_text = "3-5 days"
        else:
            life = "High"
            range_text = "5-7 days"

        suggestions = recipes.get(detected_item, {}).get(life, [])
        msg = f"Shelf Life: {life} ({range_text})"
        speak(msg)

        if suggestions:
            recipe_msg = "\nSuggested Recipes:\n- " + "\n- ".join(suggestions)
        else:
            recipe_msg = "\nNo recipes available."

        messagebox.showinfo("Shelf Life", msg + recipe_msg)
    except:
        messagebox.showerror("Error", "Invalid input or no item detected")

def clear_all():
    temp_entry.delete(0, tk.END)
    humidity_entry.delete(0, tk.END)
    panel.config(image='')
    panel.image = None

def live_predict():
    global detected_item, captured_image_path
    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        speak("Unable to connect to the live camera")
        return

    last_pred_time = time.time()
    prediction_text = "Detecting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - last_pred_time > 2:
            resized = cv2.resize(frame, IMG_SIZE)
            img_array = img_to_array(resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)[0]
            idx = np.argmax(prediction)
            label = class_labels[idx]
            confidence = prediction[idx] * 100
            prediction_text = f"{label} ({confidence:.1f}%)"
            detected_item = label
            cv2.imwrite(captured_image_path, frame)
            speak(f"Detected {label}")
            last_pred_time = time.time()

        cv2.putText(frame, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Smart Fridge - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Buttons ===
btn_frame = tk.Frame(root, bg="#f0f8ff")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Upload Image", font=("Arial", 12), command=upload_image).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Shelf Life & Recipes", font=("Arial", 12), command=predict_shelf_life).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Live Detection", font=("Arial", 12), command=live_predict).grid(row=0, column=2, padx=10)
tk.Button(btn_frame, text="Clear", font=("Arial", 12), command=clear_all).grid(row=0, column=3, padx=10)
tk.Button(btn_frame, text="Use Captured Image", font=("Arial", 12), command=use_captured_image).grid(row=0, column=4, padx=10)

root.mainloop()
