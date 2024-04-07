#GUI.py
"""A GUI built for Estimator Version 2.0"""
import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
from ModelData import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()

root.title('regression model')

root.geometry('640x480+300+300')
root.resizable(False, False)

title = tk.Label(root, text="Regression Model")
title.grid(columnspan=4)

def openfile():
    """prompts a user for a file name"""
    file_name_var.set(fd.askopenfilename())
    #print(file_name_var.get())

    csv_name = file_name_var.get()

    #make csv Data Frame global
    global csv_df
    #make variables global
    global independent_var,independent_name_var,independent_unit_var,independent_symbol_var,dependent_var,dependent_name_var,dependent_unit_var,dependent_symbol_var
    #make tkinter inp global
    global independent_inp,independent_name_inp,independent_unit_inp,independent_symbol_inp,dependent_inp,dependent_name_inp,dependent_unit_inp,dependent_symbol_inp
    #make tkinter label global
    global independent_label,independent_name_label,independent_unit_label,independent_symbol_label,dependent_label,dependent_name_label,dependent_unit_label,dependent_symbol_label

    csv_df = pd.read_csv(csv_name)

    independent_var = tk.StringVar(root)
    independent_name_var = tk.StringVar(root)
    independent_unit_var = tk.StringVar(root)
    independent_symbol_var = tk.StringVar(root)
    
    independent_label = tk.Label(root,text="select independent Variable")
    independent_name_label = tk.Label(root,text="variable name")
    independent_unit_label = tk.Label(root,text="variable unit")
    independent_symbol_label = tk.Label(root,text="variable symbol")
    independent_label.grid(row = 2, column = 0)
    independent_name_label.grid(row = 2, column = 1)
    independent_unit_label.grid(row = 2, column = 2)
    independent_symbol_label.grid(row = 2, column = 3)

    independent_opt=csv_df.columns
    independent_inp = tk.OptionMenu(root,independent_var,*independent_opt, command=lambda _ : independent_name_inp.insert(0, independent_var.get()))
    independent_name_inp = tk.Entry(root,textvariable=independent_name_var)
    independent_unit_inp = tk.Entry(root,textvariable=independent_unit_var)
    independent_symbol_inp = tk.Entry(root,textvariable=independent_symbol_var)
    independent_inp.grid(row=3,column=0)
    independent_name_inp.grid(row = 3, column = 1)
    independent_unit_inp.grid(row = 3, column = 2)
    independent_symbol_inp.grid(row = 3, column = 3)

    dependent_var = tk.StringVar(root)
    dependent_name_var = tk.StringVar(root)
    dependent_unit_var = tk.StringVar(root)
    dependent_symbol_var = tk.StringVar(root)

    dependent_label = tk.Label(root,text="select dependent Variable")
    dependent_name_label = tk.Label(root,text="variable name")
    dependent_unit_label = tk.Label(root,text="variable unit")
    dependent_symbol_label = tk.Label(root,text="variable symbol")
    dependent_label.grid(row = 4, column = 0)
    dependent_name_label.grid(row = 4, column = 1)
    dependent_unit_label.grid(row = 4, column = 2)
    dependent_symbol_label.grid(row = 4, column = 3)

    dependent_opt=csv_df.columns
    dependent_inp = tk.OptionMenu(root,dependent_var,*dependent_opt, command=lambda _ : dependent_name_inp.insert(0, dependent_var.get()))
    dependent_name_inp = tk.Entry(root,textvariable=dependent_name_var)
    dependent_unit_inp = tk.Entry(root,textvariable=dependent_unit_var)
    dependent_symbol_inp = tk.Entry(root,textvariable=dependent_symbol_var)
    dependent_inp.grid(row=5,column=0)
    dependent_name_inp.grid(row = 5, column = 1)
    dependent_unit_inp.grid(row = 5, column = 2)
    dependent_symbol_inp.grid(row = 5, column = 3)

    submit_inp['state']="normal"

file_name_var = tk.StringVar(root)
file_name_label = tk.Label(root, text="choose CSV file: ")
file_name_inp = tk.Button(root,text="Click to open file", command=openfile)
file_name_label.grid(row = 1, column = 0, columnspan=2)
file_name_inp.grid(row=1,column=2,columnspan=2)

def execute():
    X_series = csv_df[independent_var.get()]
    y_series = csv_df[dependent_var.get()]

    sample = X_series.count()

    #removes any blank elements at the end
    X_series = X_series[:sample]
    y_series = y_series[:sample]

    X = X_series.to_numpy().reshape(sample,1)
    y = y_series.to_numpy().reshape(sample,1)

    X_var = variableName(independent_name_var.get(),independent_symbol_var.get(),independent_unit_var.get())
    y_var = variableName(dependent_name_var.get(),dependent_symbol_var.get(),dependent_unit_var.get())

    mydata = data(X_var,y_var,X,y)

    graph = mydata.fit_for_tk()
    graph_img = FigureCanvasTkAgg(graph,root)
    graph_img.draw()
    graph_img.get_tk_widget().grid(row=10, columnspan=4)

    #hides labels and inputs to make room for the graph
    inp_tuple = (file_name_inp,independent_inp,independent_name_inp,independent_unit_inp,independent_symbol_inp,dependent_inp,dependent_name_inp,dependent_unit_inp,dependent_symbol_inp)

    for inp in inp_tuple:
        inp.grid_forget()

    label_tuple = (file_name_label,independent_label,independent_name_label,independent_unit_label,independent_symbol_label,dependent_label,dependent_name_label,dependent_unit_label,dependent_symbol_label)
    for label in label_tuple:
        label.grid_forget()
    
    submit_inp.grid_forget()

    detail_label = tk.Label(root, text = mydata.get_detail_string())
    detail_label.grid(columnspan=4)
    





submit_inp = tk.Button(root,text = "Execute Estimator on File",command=execute,state="disabled")
submit_inp.grid(row=100, columnspan = 4)

root.mainloop()