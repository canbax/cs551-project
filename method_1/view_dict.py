import uuid
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd


def j_tree(tree, parent, dic):
    for key in (dic.keys()):
        uid = uuid.uuid4()
        if isinstance(dic[key], dict):
            tree.insert(parent, 'end', uid, text=key)
            j_tree(tree, uid, dic[key])
        elif isinstance(dic[key], tuple):
            tree.insert(parent, 'end', uid, text=str(key) + '()')
            j_tree(tree, uid,
                   dict([(i, x) for i, x in enumerate(dic[key])]))
        elif isinstance(dic[key], list):
            tree.insert(parent, 'end', uid, text=str(key) + '[]')
            j_tree(tree, uid,
                   dict([(i, x) for i, x in enumerate(dic[key])]))
        else:
            value = dic[key]
            if isinstance(value, str):
                value = value.replace(' ', '_')
            tree.insert(parent, 'end', uid, text=key, value=value)


def tk_tree_view(data):
    # Setup the root UI
    root = tk.Tk()
    root.title("tk_tree_view")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Setup the Frames
    tree_frame = ttk.Frame(root, padding="3")
    tree_frame.grid(row=0, column=0, sticky=tk.NSEW)

    # Setup the Tree
    tree = ttk.Treeview(tree_frame, columns=('Values'))
    tree.column('Values', width=100, anchor='center')
    tree.heading('Values', text='Values')
    j_tree(tree, '', data)
    tree.pack(fill=tk.BOTH, expand=1)

    # Limit windows minimum dimensions
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
    root.mainloop()

def bin( non_unif_bin_no):
    
    train_p = pd.read_csv('data/house-prices-advanced-regression-techniques/train.csv')
    unif_bin_no = 40
    unif_split, unif_borders  = pd.cut(train_p['SalePrice'], bins = unif_bin_no, retbins=True)
    
    unif_centers = np.zeros( unif_bin_no)

    for i in range( unif_bin_no):
        unif_centers[i] =  (unif_borders[i+1] + unif_borders[i])/2
    
    a = np.zeros( unif_bin_no )

    for i in range(unif_bin_no):
        a[i] = unif_split.value_counts(sort=False)[i]
        
    labels = np.arange( non_unif_bin_no)
    
    dats, non_unif_borders  = pd.qcut(train_p['SalePrice'], q= non_unif_bin_no, labels = labels,  retbins=True)
    binned = pd.qcut(train_p['SalePrice'], q= non_unif_bin_no)
    
    for i in range( non_unif_bin_no + 1):
        x = np.zeros(2)
        x[:] = non_unif_borders[i]
        y = np.zeros(2)
        y[0] = 0
        y[1] = max(a)
        plt.plot(x,y,color='red', linewidth = 0.5)

    plt.plot(unif_centers, a,linewidth = 2)
    plt.yticks([])
    
    return dats, binned, non_unif_borders
