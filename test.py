# from two_step_zoo.datasets.generated import *
# print("Imported")
# # Get the Swiss roll dataset
# swiss_roll_train, swiss_roll_valid, swiss_roll_test = get_generated_datasets("swissroll")

# # Access the data and labels
# swiss_roll_train_data, swiss_roll_train_labels = swiss_roll_train.x.numpy(), swiss_roll_train.y.numpy()
# swiss_roll_valid_data, swiss_roll_valid_labels = swiss_roll_valid.x.numpy(), swiss_roll_valid.y.numpy()
# swiss_roll_test_data, swiss_roll_test_labels = swiss_roll_test.x.numpy(), swiss_roll_test.y.numpy()

# # Print the shapes of the datasets
# print("Swiss Roll Train Data Shape:", swiss_roll_train_data.shape)
# print("Swiss Roll Train Labels Shape:", swiss_roll_train_labels.shape)

# print("Swiss Roll Valid Data Shape:", swiss_roll_valid_data.shape)
# print("Swiss Roll Valid Labels Shape:", swiss_roll_valid_labels.shape)

# print("Swiss Roll Test Data Shape:", swiss_roll_test_data.shape)
# print("Swiss Roll Test Labels Shape:", swiss_roll_test_labels.shape)

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
# from mpl_toolkits.mplot3d import Axes3D

# # Function to plot 3D Swiss roll
# def plot_swiss_roll(data, labels, title):
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')

#     ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=20)

#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Z-axis')
#     ax.set_title(title)

#     plt.show()

# # Plot the training dataset
# plot_swiss_roll(swiss_roll_train_data, swiss_roll_train_labels, 'Swiss Roll - Training Dataset')

# # Plot the validation dataset
# plot_swiss_roll(swiss_roll_valid_data, swiss_roll_valid_labels, 'Swiss Roll - Validation Dataset')

# # Plot the test dataset
# plot_swiss_roll(swiss_roll_test_data, swiss_roll_test_labels, 'Swiss Roll - Test Dataset')

############################################################

from two_step_zoo.id_estimator.estimator import MLEIDEstimator
from two_step_zoo.datasets.generated import *
import numpy as np
from sklearn.utils import shuffle as util_shuffle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

print("Imported")

# Function to plot 3D dataset
def plot_3d_dataset(data, labels, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=20)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(title)

    plt.show()

# Generate sphere and annulus dataset
sphere_annulus_train_data, sphere_annulus_train_labels = generate_sphere_annulus_dataset(size=10000, sphere_offset=[0.0, 0.0, 0.0], annulus_offset=[2.0, 0.0], annulus_radius=1.0)
sphere_annulus_valid_data, sphere_annulus_valid_labels = generate_sphere_annulus_dataset(size=1000, sphere_offset=[0.0, 0.0, 0.0], annulus_offset=[2.0, 0.0], annulus_radius=1.0)
sphere_annulus_test_data, sphere_annulus_test_labels = generate_sphere_annulus_dataset(size=5000, sphere_offset=[0.0, 0.0, 0.0], annulus_offset=[2.0, 0.0], annulus_radius=1.0)

# Print the shapes of the datasets
print("Sphere and Annulus Train Data Shape:", sphere_annulus_train_data.shape)
print("Sphere and Annulus Train Labels Shape:", sphere_annulus_train_labels.shape)

print("Sphere and Annulus Valid Data Shape:", sphere_annulus_valid_data.shape)
print("Sphere and Annulus Valid Labels Shape:", sphere_annulus_valid_labels.shape)

print("Sphere and Annulus Test Data Shape:", sphere_annulus_test_data.shape)
print("Sphere and Annulus Test Labels Shape:", sphere_annulus_test_labels.shape)

# Plot the training dataset
plot_3d_dataset(sphere_annulus_train_data.numpy(), sphere_annulus_train_labels.numpy(), 'Sphere and Annulus - Training Dataset')

# Plot the validation dataset
plot_3d_dataset(sphere_annulus_valid_data.numpy(), sphere_annulus_valid_labels.numpy(), 'Sphere and Annulus - Validation Dataset')

# Plot the test dataset
plot_3d_dataset(sphere_annulus_test_data.numpy(), sphere_annulus_test_labels.numpy(), 'Sphere and Annulus - Test Dataset')

cluster_cfg = {
    "num_clusters": 10,
    "latent_dim": 64,
    "id_estimator": {
        "id_estimates_save": "here.pth",
        # Add other ID estimator parameters if needed
    },
    # Add other clustering configuration parameters
}

# Create an instance of MLEIDEstimator
estimator = MLEIDEstimator(cluster_cfg, writer=None)  # Pass appropriate writer if available

# Get ID estimates
id_estimates = estimator.get_id_estimates([sphere_annulus_train_data, sphere_annulus_valid_data, sphere_annulus_test_data])

# Print the estimated intrinsic dimensions
print("Estimated Intrinsic Dimensions:", id_estimates)


#################################################

