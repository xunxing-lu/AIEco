import cv2
import openai
import os

def capture_image():
    """Capture an image using the camera."""
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None

    print("Press 's' to capture an image or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print(f"Image saved to {image_path}")
            break
        elif key == ord('q'):
            print("Exiting without capturing an image.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return image_path if 'image_path' in locals() else None

def send_image_to_llm(image_path):
    """Send the captured image to the LLM for analysis."""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    try:
        with open(image_path, "rb") as image_file:
            response = openai.Image.create(
                model="gpt-4-vision-preview",
                file=image_file
            )

        print("LLM Response:")
        print(response)
    except Exception as e:
        print(f"Error sending image to LLM: {e}")

def main():
    """Main function to capture an image and send it to the LLM."""
    print("Starting Vision LLM Application...")
    image_path = capture_image()
    if image_path:
        send_image_to_llm(image_path)

if __name__ == "__main__":
    main()