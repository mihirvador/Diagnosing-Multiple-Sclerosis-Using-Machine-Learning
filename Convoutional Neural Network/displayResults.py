import pandas as pd
import matplotlib.pyplot as plt

def lossCurve(history):
    display = pd.DataFrame(history)
    plt.plot(display["loss"])
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.savefig("Gen 2/Code/CNN/Custom/Visualizations/lossHistory.png")
    print("Displaying Loss")