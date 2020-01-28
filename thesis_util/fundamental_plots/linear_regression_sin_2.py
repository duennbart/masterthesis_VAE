#%%
import matplotlib.pyplot as plt
import tikzplotlib
#import matplotlib2tikz
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
#%%

# generate random data-set
np.random.seed(0)
x = np.random.rand(50, 1)*20
y =np.sin(x) #+ 0.1 * np.random.rand(50, 1)
degree = 3
#%%
# Create linear regression object

poly_model = make_pipeline(PolynomialFeatures(degree),linear_model.LinearRegression())

#%%
xfit = np.linspace(0, 20, 100)
# Train the model using the training sets
poly_model.fit(x, y)
#%%
# Make predictions using the testing set
y_pred = poly_model.predict(xfit[:, np.newaxis])

#%%
# Plot outputs
f = plt.figure(figsize=(8, 6))
ax = f.add_subplot(111)
ax.scatter(x, y,  color='black')
ax.plot(xfit, y_pred, color='blue', linewidth=3)
plt.style.use("ggplot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim([-1.1,1.1])
#ax.set_title("Simple linear regression model  $y=w_1 x + b$")
ax.grid(True)
#ax.text(0.1, 0.9,'$w_1 =' +'{:.2f}'.format(w) +'$ \n $b ='+'{:.2f}'.format(b) +'$', ha='center', va='center',transform=ax.transAxes,  bbox=dict(facecolor='none', edgecolor='black'))
#plt.xticks(())
#plt.yticks(())
tikzplotlib.save("C:\\Users\\ga45tis\\GIT\\masterthesisgeneral\\latex\\900 Report\\images\\fundamentels\\sin_linear_regression_poly3.tex")
plt.show()