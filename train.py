import matplotlib.pyplot as plt
history=my_model.fit(X_train,y_train, batch_size = 512,epochs = 100,shuffle=True,validation_data=(X_val,y_val))  ## Removing the shuffle cause shuffle seems to undo the CV
print(history.history['loss']) #batch 1028
print(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()