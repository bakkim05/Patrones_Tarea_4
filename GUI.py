import tkinter as tk

class window:
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Introduccion al Reconocimiento de Patrones - Tarea 4")
        self.root.minsize(200,50)

        self.svm_button = tk.Button(self.root, text="SVM",  width = 100, command=self.svm)
        self.svm_button.pack()

        self.krs_button = tk.Button(self.root, text="Keras",  width = 100, command=self.krs)
        self.krs_button.pack()

        self.close_button = tk.Button(self.root, text="Close",  width = 100, command=self.quit)
        self.close_button.pack()



    def svm(self):
        print("Aqui va la funcion de Support Vector Machine")
        self.root.mainloop()

    def krs(self):
        print("Aqui va la funcion de Keras")
        self.root.mainloop()

    def quit(self):
        self.root.destroy()


wndw = window()
