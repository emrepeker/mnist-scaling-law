
    
#     rows = []
#     for i in range(len(subset)):
#         img, label = subset[i]
#         img = img.view(-1).numpy() # Flatten 28x28 -> 784
#         rows.append([label]+ img.tolist()) # [0,1] since ToTensor divides by 255
#     df = pd.DataFrame(rows)
#     df.to_csv(f"data/mnist_train_{size}.csv", index = False)
#     print(f"Saved data/mnist_train_{size}.csv with {