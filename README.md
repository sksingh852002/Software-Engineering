# Documentation for Real-Time Handwriting Recognition and AI Response System

## Overview

This project implements a **Real-Time Handwriting Recognition and AI Response System** using computer vision and AI. The application captures hand gestures through a webcam feed, processes them using the cvzone HandTrackingModule, and interprets gestures to draw on a virtual canvas. The canvas content is processed using Google’s Generative AI (Gemini-1.5-flash) to generate answers based on the user's drawings or gestures. The application is built using **Python** and **Streamlit** for creating a web interface.

---

## Key Features

1. **Hand Gesture Recognition**: Uses cvzone and MediaPipe to track hand gestures and interpret them.
2. **Real-Time Drawing**: Allows users to draw mathematical expressions on a virtual canvas using hand gestures.
3. **Generative AI Integration**: Sends the canvas image to Google’s Generative AI model to solve and respond with results.
4. **Streamlit Interface**: Provides a user-friendly interface with real-time webcam feed and answer display.

---

## System Requirements

- **Python 3.10 or higher**
- **Webcam**
- **Dependencies** (see `requirements.txt`):
  - pip\~=24.3.1
  - attrs\~=24.2.0
  - pillow\~=11.0.0
  - tornado\~=6.4.2
  - Jinja2\~=3.1.4
  - cvzone\~=1.6.1
  - opencv-python\~=4.10.0.84
  - mediapipe\~=0.10.18
  - numpy\~=1.26.4
  - Flask\~=3.1.0
  - Google Generative AI (Gemini-1.5-flash) SDK

  Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```plaintext
.
├── app.py             # Main application code
├── requirements.txt   # Dependency list
└── README.md          # Documentation
```

---

## Implementation Details

### 1. **User Interface**

The interface is created using **Streamlit**:

- **Title Section**: Displays the project title, team members, and guide details.
- **Webcam Feed**: Captures live video feed and displays it in real time.
- **Answer Section**: Displays the AI’s response based on the canvas content.

#### Code for Interface:

```python
st.markdown("<h1 style='color:#e9c46a;'>Virtual Math Calculator</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#e9c46a;'>Saurav Kumar (2230283), Shravan Yadav(2230290), Shubham Kumar(2230292)</h3>", unsafe_allow_html=True)


col1, col2 = st.columns([2, 1])  # Creating two columns with a 2:1 ratio

with col1:
    run = st.checkbox('Run', value=True)  # Checkbox to start/stop the webcam feed
    FRAME_WINDOW = st.image([])  # Placeholder for webcam feed

with col2:
    output_text_area = st.title("Answer")  # Title for the answer section
    output_text_area = st.markdown("", unsafe_allow_html=True)  # Placeholder for displaying the answer
```

---

### 2. **Hand Gesture Recognition**

This module uses **cvzone** and **MediaPipe** to detect hand gestures and track finger movements.

- **HandDetector Initialization**:

```python
detector = HandDetector(
    staticMode=False,
    maxHands=2,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)
```

- **Hand Information Extraction**:
  The function `getHandInfo()` processes each webcam frame to detect hand landmarks and determine the status of fingers (up/down).

```python
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None
```

---

### 3. **Drawing on the Virtual Canvas**

The **index finger** is used to draw, while the **thumb** is used to clear the canvas. A red line is drawn to represent the strokes.

- **Drawing Logic**:

```python
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = tuple(lmList[8][0:2])
        if prev_pos is None:
            prev_pos = current_pos
        else:
            current_pos = (
                (current_pos[0] + prev_pos[0]) // 2,
                (current_pos[1] + prev_pos[1]) // 2
            )
        cv2.line(canvas, current_pos, prev_pos, color=(0, 0, 255), thickness=10)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up
        canvas = np.zeros_like(img)  # Clear canvas
    return current_pos, canvas
```

---

### 4. **Google Generative AI Integration**

The canvas content is sent to Google’s Generative AI model for processing mathematical expressions.

- **Model Initialization**:

```python
genai.configure(api_key="<YOUR_API_KEY>")  # API Key for Generative AI
model = genai.GenerativeModel("gemini-1.5-flash")
```

- **Sending Data to AI**:

```python
def sendToAI(model, canvas, fingers):
    if fingers == [0, 1, 1, 1, 1]:  # Four fingers up
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["What is", pil_image])
        return response.text
```

---

### 5. **Webcam and Canvas Integration**

The webcam feed and the canvas are combined using OpenCV functions for display.

```python
while run:
    success, img = cap.read()  # Capture frame
    img = cv2.flip(img, flipCode=1)  # Flip horizontally
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels='BGR')

    if output_text:
        output_text_area.markdown(f"<span style='color:#e9c46a;'>{output_text}</span>", unsafe_allow_html=True)
    cv2.waitKey(1)
```

---

### 6. **Application Workflow**

1. **Start Webcam**: The webcam initializes and displays a real-time feed.
2. **Hand Detection**: Detects hand gestures using cvzone and MediaPipe.
3. **Drawing**: Allows users to draw on a virtual canvas using gestures.
4. **AI Processing**: Sends canvas content to Google’s Generative AI for solving.
5. **Output Display**: Shows the AI’s response on the Streamlit interface.

---

## Setup Instructions


### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Run the Application**

```bash
streamlit run app.py
```

---

## Future Enhancements

1. **Gesture Customization**: Add support for more complex gestures to perform specific operations.
2. **Enhanced AI Integration**: Allow natural language queries in addition to mathematical expressions.
3. **Improved UI**: Add controls for canvas color, line thickness, and save functionality.

---

## Conclusion

This project demonstrates the integration of computer vision and AI to create an innovative and interactive application. The **Real-Time Handwriting Recognition and AI Response System** leverages hand tracking, real-time drawing, and generative AI to provide a seamless user experience for solving problems visually.


