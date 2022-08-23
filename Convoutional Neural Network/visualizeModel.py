import tensorflow as tf

model = tf.keras.models.load_model("Gen 2/Code/CNN/Custom/bestClassification.h5")
modelpng = "Gen 2/Code/CNN/Custom/Visualizations/model.png"
tf.keras.utils.plot_model(model, to_file=modelpng, show_shapes=True)
print("done")

