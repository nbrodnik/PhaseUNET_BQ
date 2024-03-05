import numpy as np
import matplotlib.pyplot as plt

import support as supp

plt.rcParams.update({"font.size": 20})

path = "./demo_data/TestBFCEG1_history.npy"

data = np.load(path, allow_pickle=True).item()

loss = data["loss"]
val_loss = data["val_loss"]

fig, axl = plt.subplots(1, 1, num=1, figsize=(10, 7))
axl.plot(np.arange(len(loss))+1, loss, label="Training")
axl.plot(np.arange(len(loss))+1, val_loss, label="Validation")
axl.set_xlabel("Epoch", fontsize=20)
axl.set_ylabel("Loss", fontsize=20)
axl.legend(fontsize=16)
plt.tight_layout()
plt.show()