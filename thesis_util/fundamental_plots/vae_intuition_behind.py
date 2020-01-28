#%%
import matplotlib.pyplot as plt
import numpy as np

random_variable_z_x = np.random.normal(0,1,20)
random_variable_z_y = np.random.normal(0,1,20)
plt.plot(random_variable_z_x,random_variable_z_y, 'ro')
plt.axis('off')
plt.show()

#%%
random_variable_z_x = random_variable_z_x/10 + random_variable_z_x/np.linalg.norm(random_variable_z_x)
random_variable_z_y = random_variable_z_y/10 + random_variable_z_y/np.linalg.norm(random_variable_z_y)
plt.plot(random_variable_z_x, random_variable_z_y, 'ro')
plt.axis('off')
plt.show()

#%%
# Create and plot multivariate normal distribution
mean = [0, 0]
cov = [[1,0],[0,1]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
plt.figure(1)
plt.plot(x, y, 'ro')
plt.axis('equal')
plt.savefig(fname='vae_intuition_explained1.svg')
plt.show()

#%%
# Generate z
def g(xy):
    res_z = []
    for z in xy:
        z = np.array(z)
        res_z.append(z / 10 + z / np.linalg.norm(z))
    return res_z
xy = zip(x, y)
res_z = g(xy)

# Plot z
zx, zy = zip(*res_z)
plt.figure(2)
plt.plot(zx, zy, 'ro')
plt.axis('equal')
plt.savefig(fname='vae_intuition_explained2.pdf')
plt.show()