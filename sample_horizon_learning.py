"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

# Import tf if using backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
import deepxde as dde
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.graph_objects as go

def generate_evolution_model(domain, boundary, width, depth, rate, epochs):

    def pde(x, y):
        # y = (f)
        # x = (r,t)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        return dy_t - ((1-(1/(2*x[:,0:1])))/((1+ (1/(2*x[:,0:1]))**3)))


    main_domain = dde.geometry.Rectangle([0.1,0],[1,100])
    geom = main_domain


    ic = dde.icbc.DirichletBC(
        geom,
        lambda x: x[:,0:1] - 0.51,
        lambda x, on_boundary:  np.isclose(x[1],100),
    )

#     dic = dde.icbc.OperatorBC(
#     geom,
#     lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
#     lambda _, on_initial: on_initial,)


    data = dde.data.PDE(
        geom, pde, [ic], num_domain=domain, num_boundary=boundary)


    net = dde.nn.FNN([2] + [width] * depth + [1], "tanh", "Glorot normal")


    model = dde.Model(data, net)
    model.compile("L-BFGS", lr=rate)
    losshistory, train_state = model.train(iterations=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)



    # Generate uniformly spaced points within the specified ranges
    num_points = 1000

    # For the first value (0 to 1 range)
    first_value_range = np.linspace(0, 1, num_points)

    # For the second value (0 to 100 range)
    second_value_range = np.linspace(0, 100, num_points)

    # Create a meshgrid of all possible combinations of first and second values
    first, second = np.meshgrid(first_value_range, second_value_range)

    # Combine the first and second values to form the final input data
    uniform_data = np.column_stack((first.ravel(), second.ravel()))

    # Assuming you have already computed 'Z' using the model.predict() function
    # If not, replace 'Z' with the actual predictions obtained from the model

    Z = model.predict(uniform_data)


    # Reshape 'Z' to match the shape of the meshgrid
    Z = Z.reshape(first.shape)

    # Create an interactive 3D surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=first, y=second)])

    # Add labels to the axes
    fig.update_layout(scene=dict(xaxis_title='r',
                                 yaxis_title='t',
                                 zaxis_title='f'))

    print("This is the learned solution: ")
    # Show the plot
    fig.show()




    # Assuming you already have 'uniform_data' as mentioned in your previous code.

    # Define a function that takes the input data and returns the output value.
    def analytic_function(input_data):
        # Assuming the input_data has two columns, representing first and second values.
        r = input_data[:, 0]
        t = input_data[:, 1]

        # Calculate the sum of the first and second values.
        f = (8 * (-0.0125 - 0.134375 * r + 5.6875 * r**2 - 0.0625 * t * r**2 + 23.9375 * r**3 -
         0.25 * t * r**3 - 0.5 * r**4 - 98.65 * r**5 +  t * r**5 - 97.8 * r**6 +
          t * r**6 + 1 * r**7)) / ((1 + 2 * r)**3 * (0.125 + 0.75 * r + 1.5 * r**2 +  r**3))

        return f


    analytic_f = analytic_function(uniform_data)
    analytic_f = analytic_f.reshape(first.shape)
    abs_deviation = abs(Z-analytic_f)


     # Create an interactive 3D surface plot
    fig = go.Figure(data=[go.Surface(z=analytic_f, x=first, y=second)])

    # Add labels to the axes
    fig.update_layout(scene=dict(xaxis_title='r',
                                 yaxis_title='t',
                                 zaxis_title='f'))

    print("This is the analytic solution:")
    # Show the plot
    fig.show()

     # Create an interactive 3D surface plot
    fig = go.Figure(data=[go.Surface(z=abs_deviation, x=first, y=second)])

    # Add labels to the axes
    fig.update_layout(scene=dict(xaxis_title='r',
                                 yaxis_title='t',
                                 zaxis_title='f'))

    print("This is the deviation:")
    # Show the plot
    fig.show()







    # Given data
    num_points = 1000
    r_points = np.linspace(0, 1, num_points)
    t_points = np.linspace(0, 100, num_points)
    r, t = np.meshgrid(r_points, t_points)
    uniform_data = np.column_stack((r.ravel(), t.ravel()))

    # Assuming you have already computed 'Z' using the model.predict() function
    # If not, replace 'Z' with the actual predictions obtained from the model
    Z = model.predict(uniform_data)

    # Find the index of the entry in Z that is nearest to zero
    nearest_index = np.argmin(np.abs(Z))

    # Find the corresponding r, t, and Z values using the index
    nearest_r = r.ravel()[nearest_index]
    nearest_t = t.ravel()[nearest_index]
    nearest_Z = Z[nearest_index]

    print("Nearest r:", nearest_r)
    print("Nearest t:", nearest_t)
    print("Corresponding f:", nearest_Z)
    print("f at true horizon: ", model.predict([[0.5,0]]))




    return model



model = generate_evolution_model(1000, 1000, 200, 2, 0.01, 50000)