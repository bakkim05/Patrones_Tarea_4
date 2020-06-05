from tkinter import *
from tkinter import ttk, colorchooser, filedialog
from PIL import Image
#from PIL import ImageGrab
import pyscreenshot as ImageGrab
import numpy as np
from joblib import load
from keras.models import load_model

class main:
    def __init__(self,master):
        self.master = master
##        self.color_fg = 'black'
##        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 15
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)
        self.c.bind('<ButtonRelease-1>',self.reset)

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill="black",capstyle=ROUND,smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):
        self.old_x = None
        self.old_y = None      

##    def changeW(self,e):
##        self.penwidth = e

    def save(self):
        file = filedialog.asksaveasfilename(filetypes=[('Portable Network Graphics','*.png')])
        if file:
            x = self.master.winfo_rootx() + self.c.winfo_x()
            y = self.master.winfo_rooty() + self.c.winfo_y()
            x1 = x + self.c.winfo_width()
            y1 = y + self.c.winfo_height()

            ImageGrab.grab().crop((x,y,x1,y1)).resize((28,28)).save(file)
            
            
           

    def clear(self):
        self.c.delete(ALL)


    def svm(self):
        print("Aqui va la funcion de Support Vector Machine")
        ima = Image.open("iden.png")
        im = ima.convert('1')
        a=np.asarray(im,dtype=np.float32)
        a=a / 255.0 * 2 - 1
        a=a.reshape(1,784)
        clf = load('rbf-001-28.joblib')
        val = clf.predict(a)
        print(val)
        self.master.mainloop()

    def krs(self):
        print("Aqui va la funcion de Keras")
        ima = Image.open("iden.png")
        im = ima.convert('1')
        a=np.asarray(im,dtype=np.float32)
        a=a.reshape(1,784)
        clf = load_model('network.h5')
        val = clf.predict(a)
        print(np.argmax(val))
        self.master.mainloop()
    


    def drawWidgets(self):
        
        self.c = Canvas(self.master,width=220,height=220,bg='white',)
        self.c.pack(fill=BOTH,expand=True)

        self.clear_button = Button(self.master, text="Reset",  width = 30, command=self.clear)
        self.clear_button.pack()

        self.save_button = Button(self.master, text="Save",  width = 30, command=self.save)
        self.save_button.pack()

        self.svm_button = Button(self.master, text="SVM",  width = 30, command=self.svm)
        self.svm_button.pack()

        self.krs_button = Button(self.master, text="Keras",  width = 30, command=self.krs)
        self.krs_button.pack()

        self.close_button = Button(self.master, text="Close",  width = 30, command=self.master.destroy)
        self.close_button.pack()

        

##        menu = Menu(self.master)
##        self.master.config(menu=menu)
##        filemenu = Menu(menu)
##        menu.add_cascade(label='File..',menu=filemenu)
##        filemenu.add_command(label='Save..',command=self.save)
##
##        optionmenu = Menu(menu)
##        menu.add_cascade(label='Options',menu=optionmenu)
##        optionmenu.add_command(label='Clear Canvas',command=self.clear)
##        optionmenu.add_command(label='Exit',command=self.master.destroy) 
        
        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Tarea 4')
    root.resizable(False,False)
    root.mainloop()
