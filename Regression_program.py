# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 23:39:03 2022

@author: Garduno
"""

import tkinter as tk
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn import linear_model
from TkinterDnD2 import DND_FILES, TkinterDnD
from pathlib import Path

def plot_graph(frame_plot,Combo_x,Combo_y,df):
    global m
    global b
    global data
    for widget in frame_plot.winfo_children():
        widget.destroy()
    figure = plt.Figure(figsize = (6,5.85))
    ax = figure.add_subplot(111)
    X = Combo_x.get()
    Y = Combo_y.get()
    ax.scatter(df[X],df[Y], color = 'g',s=0.25)
    scatter = FigureCanvasTkAgg(figure, frame_plot) 
    scatter.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH)
        
        
    xd=np.array(df[X])
    yd=np.array(df[Y])
    m = 0
    b = 0
    alpha = 0.005
    #print(max(df[Y]))
    #print(min(df[Y]))
    iterations = 10000
    n = len(xd)
    for i in range(iterations):
        prediction = (m*xd)+b
        m = m - (alpha*((1/n)*np.sum((prediction-yd)*xd)))             
        b = b - (alpha*((1/n)*np.sum(prediction-yd)))                     
    prediction = (m*xd) + b
    ax.plot(xd, prediction, color='red',linewidth = 3)
        
    regr = linear_model.LinearRegression()
    regr.fit(np.array(df[X]).reshape(-1, 1), np.array(df[Y]).reshape(-1, 1))
    Y_pred = regr.predict(np.array(df[X]).reshape(-1, 1))
    ax.plot(np.array(df[X]).reshape(-1, 1), Y_pred, color="blue", linewidth=2)
        
    ax.set_xlabel(X)
    ax.set_ylabel(Y)

def prediction(T,entry):
    global m
    global b
    global data
    data = float(entry.get())
    result = (m*data)+b
    T.delete('1.0', END)
    T.insert(tk.END, "m={:.2f}, b={:.2f}, x={}\n(m*(x))+b = y\n({:.2f}*({:.2f}))+{:.2f} = {:.2f}".format(float(m),float(b),float(data),float(m),float(data),float(b),float(result)))
    
class Regression:
    def __init__(self):
        global m
        global b
        global data
        root = TkinterDnD.Tk()
        #root.geometry('800x800')
        #root.attributes("-fullscreen", True)
        root.state('zoomed')
        root.configure(bg = 'red')
        root.title('Regresiòn')
        ttk.Button(root, text='Exit', 
                   command=root.destroy).pack(side=tk.BOTTOM)
        
        
        self.df = pd.DataFrame()
        frame = Frame(root, width=400,height=100) 
        frame.pack()
        self.my_tree = ttk.Treeview(frame)
        scroll_Y = tk.Scrollbar(frame, orient="vertical", command=self.my_tree.yview)
        scroll_X = tk.Scrollbar(frame, orient="horizontal", command=self.my_tree.xview)
        self.my_tree.configure(yscrollcommand=scroll_Y.set, xscrollcommand=scroll_X.set)
        scroll_Y.pack(side="right", fill="y")
        scroll_X.pack(side="bottom", fill="x")
        
        self.Options_frames=Frame(root, width=200,height=100)
        self.Options_frames.pack(side=tk.TOP)
        self.Prediction_frames=Frame(root, width=200,height=100)
        self.Prediction_frames.pack(side=tk.TOP)
        label_value = tk.Label(self.Prediction_frames,text = "X Value")
        label_value.pack(side=tk.TOP, fill=tk.BOTH)
        self.entry = tk.Entry (self.Prediction_frames)
        self.entry.pack(side=tk.TOP, fill=tk.BOTH)
        ttk.Button(self.Prediction_frames, text='Prediction', 
                   command=lambda:prediction(self.T,self.entry)).pack(side=tk.TOP)
        self.T = Text(self.Prediction_frames, height = 3, width = 52)
        self.T.pack(side = tk.TOP)
        
        
        
        frame.drop_target_register(DND_FILES)
        frame.dnd_bind("<<Drop>>", self.drop_file)
        self.frame_plot = Frame(root, width=480,height=320) 
        self.frame_plot.pack(side=tk.BOTTOM, fill=tk.BOTH)
        root.mainloop() 
    
    def drop_file(self,event):
        file_paths = self._parse_drop_files(event.data)
        for file_path in file_paths:
            if file_path.endswith(".csv"):
                path_object = Path(file_path)
                file_name = path_object.name
                self.df = pd.read_csv (file_name)
                for widget in self.Options_frames.winfo_children():
                    widget.destroy()
                for widget in self.frame_plot.winfo_children():
                    widget.destroy()
                combo_frame_x = Frame(self.Options_frames, width=200,height=100)
                combo_frame_x.pack(side=tk.LEFT)
                label_x = tk.Label(combo_frame_x,text = "X axis")
                label_x.pack(side=tk.LEFT, fill=tk.BOTH)
                self.Combo_x = ttk.Combobox(combo_frame_x,values=list(self.df.columns))
                self.Combo_x.current(1)
                self.Combo_x.pack(side=tk.LEFT, fill=tk.BOTH)
        
                combo_frame_y = Frame(self.Options_frames, width=200,height=100)
                combo_frame_y.pack(side=tk.LEFT)
                label_y = tk.Label(combo_frame_y,text = "Y axis")
                label_y.pack(side=tk.LEFT, fill=tk.BOTH)
                self.Combo_y = ttk.Combobox(combo_frame_y,values=list(self.df.columns))
                self.Combo_y.current(1)
                self.Combo_y.pack(side=tk.LEFT, fill=tk.BOTH)
                plot_button = ttk.Button(self.Options_frames,text="m,b and plot", command=lambda:plot_graph(self.frame_plot,self.Combo_x,self.Combo_y,self.df)).pack(side=tk.LEFT)
                self.my_tree.stored_dataframe = self.df
                self.my_tree.delete(*self.my_tree.get_children())
                columns = list(self.df.columns)
                self.my_tree.__setitem__("column", columns)
                self.my_tree.__setitem__("show", "headings")

        for col in columns:
            self.my_tree.heading(col, text=col)

        df_rows = self.df.to_numpy().tolist()
        for row in df_rows:
            self.my_tree.insert("", "end", values=row)
        self.my_tree.pack(pady = 1)
    
    
    def _parse_drop_files(self, filename):
        size = len(filename)
        res = []  # list of file paths
        name = ""
        idx = 0
        while idx < size:
            if filename[idx] == "{":
                j = idx + 1
                while filename[j] != "}":
                    name += filename[j]
                    j += 1
                res.append(name)
                name = ""
                idx = j
            elif filename[idx] == " " and name != "":
                res.append(name)
                name = ""
            elif filename[idx] != " ":
                name += filename[idx]
            idx += 1
        if name != "":
            res.append(name)
        return res        
    
m = 0 
b = 0
data = 0        
ventana = Regression()