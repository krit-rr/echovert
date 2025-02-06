import tkinter as tk
from tkinter import messagebox
from train_model import train_model
from train_model import clean_data

class HateSpeechDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Hate Speech Detector")
        self.root.geometry("500x200")

        self.model, self.vectorizer, self.report, self.accuracy = train_model()

        self.label = tk.Label(root, text="Enter text to check:")
        self.label.pack(pady=5)

        self.text_input = tk.Text(root, height=5, width=50)
        self.text_input.pack(pady=5)

        self.check_button = tk.Button(root, text="Check Text", command=self.check_text)
        self.check_button.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=5)

    def check_text(self):
        if not self.model:
            messagebox.showerror("Error", "No model loaded! Please load a trained model.")
            return

        user_input = self.text_input.get("1.0", tk.END).strip()
        user_input = clean_data(user_input)
        if user_input:
            text_vector = self.vectorizer.transform([user_input])  # Vectorize input
            prediction = self.model.predict(text_vector)[0]
            print(text_vector)
            print(self.model.predict(text_vector))
            if prediction == 0:
                result = "Hate speech detected! You should consider rephrasing it."
            elif prediction == 1:
                result = "Offensive language detected! You should consider rephrasing it."
            else:
                result = "Your text seems clean, feel free to post it!"
            self.result_label.config(text=result, fg="red" if (prediction == 0 or prediction == 1)
                                    else "green")
        else:
            self.result_label.config(text="Please enter some text.", fg="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = HateSpeechDetector(root)
    root.mainloop()