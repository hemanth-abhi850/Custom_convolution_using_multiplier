model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

test_img_value = 56
image = X_test[test_img_value]

# Display the selected image
plt.imshow(image.squeeze(), cmap='gray')
plt.show()

# Reshape the image for prediction (CNN expects batch dimension)
image = image.reshape(1, 28, 28, 1)  # Add batch dimension

# Predict the class
predicted_label = model.predict(image)
predicted_digit = np.argmax(predicted_label)  # Get the class with highest probability
confidence = np.max(predicted_label) * 100

print(f"Predicted Digit: {predicted_digit}")
print(f"Actual Label: {y_test[test_img_value]}")
print(f"Confidence: {confidence:.2f}%")
