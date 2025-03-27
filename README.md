## Overview

This project implements a **Real-Time Handwriting Recognition and AI Response System** using computer vision and AI. The application captures hand gestures through a webcam feed, processes them using the cvzone HandTrackingModule, and interprets gestures to draw on a virtual canvas. The canvas content is processed using Google’s Generative AI (Gemini-1.5-flash) to generate answers based on the user's drawings or gestures. The application is built using **Python** and **Streamlit** for creating a web interface.

---

## Key Features

1. **Hand Gesture Recognition**: Uses cvzone and MediaPipe to track hand gestures and interpret them.
2. **Real-Time Drawing**: Allows users to draw mathematical expressions on a virtual canvas using hand gestures.
3. **Generative AI Integration**: Sends the canvas image to Google’s Generative AI model to solve and respond with results.
4. **Streamlit Interface**: Provides a user-friendly interface with real-time webcam feed and answer display.

---
