import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'real_time'))
from real_time.real_time_detection import SignLanguageInterpreter




def main():
    interpreter = SignLanguageInterpreter()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Get prediction 
        predicted_char, confidence = interpreter.predict(frame)

        # Display results
        if predicted_char:
            cv2.putText(frame, f"Sign: {predicted_char} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0., 255, 0), 2)
        cv2.imshow('Sign Language Interpreter', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


            