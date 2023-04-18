import tkinter as tk


def window():
    window = tk.Tk()
    ###WIDGETS
    ##LABELS
    label = tk.Label(
        text="Hello Tkinter", foreground="white", background="black", width=10, height=10
    )
    label.pack()
    ###ENTRY
    entry = tk.Entry(fg="red", bg="orange", width=10)
    entry.pack()
    window.mainloop()
    #name=entry.get()
    entry.insert(tk.END, " John")
    ###BUTTONS
    button = tk.Button(text="Click here!", fg="blue", bg="orange", width=10, height=2)
    button.pack()
    window.mainloop()
