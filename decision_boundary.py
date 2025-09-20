import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(clf, X, y, cmap='RdYlBu', device='cpu'):
    h = 0.01

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Convert meshgrid points to torch tensor
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).float().to(device)

    # Model predictions
    with torch.no_grad():
        outputs = clf(grid_tensor)  # shape [num_points, num_classes]
        if outputs.shape[1] > 1:   # multi-class
            Z = outputs.argmax(dim=1)
        else:                       # binary classification
            Z = (outputs > 0.5).long()
    Z = Z.cpu().numpy().reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=cmap, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()
