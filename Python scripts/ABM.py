import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd

def vectorized():
    n_a = 100
    dt = 0.001
    t_0 = 0
    T = 1e+2 #1e+4
    sigma = 0.7 # 1.0

    dx = 0.1
    dy = dx
    x_min = -2.0
    x_max = 2.0
    y_min = -1.5
    y_max = 1.5

    nx, ny = (int((x_max-x_min)/dx), int((y_max-y_min)/dy))
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    V = (X**2-1)**2 + 3.5*Y**2
    
    V_x = lambda x,y: 4*x*(x**2-1)
    V_y = lambda x,y: 7*y

    x_0 = np.random.uniform(x_min, x_max, n_a)
    y_0 = np.random.uniform(y_min, y_max, n_a)

    initial_positions = np.array([x_0,y_0])
    x_t = initial_positions
    file_nbr = 0
    for i, t in enumerate(np.linspace(t_0, T, int((T-t_0)/dt))):
        x = x_t[0,:]
        y = x_t[1,:]
        grad_x_V = V_x(x,y) #(V(x+dx,y) - V(x-dx,y))/(2*dx) 
        grad_y_V = V_y(x,y) #(V(x,y+dy) - V(x,y-dy))/(2*dy) 
        grad_V = np.array([grad_x_V, grad_y_V])
        x_t_plus_1 = x_t - grad_V * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (2,n_a))
        x_t = x_t_plus_1

        if i % 100 == 0 and i != 0:
            print(np.min(x_t_plus_1))
            print(np.max(x_t_plus_1))

            fig = plt.figure()
            plt.scatter(x_t[0,:], x_t[1,:])
            plt.contour(X,Y,V)
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("simulations/simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1

            if file_nbr == 10:
                break

        
    nbr_files = file_nbr
    images = []
    for i in range(nbr_files):
        filename = "simulations/simulation_nbr_" + str(i) + ".png"
        images.append(imageio.imread(filename))
    imageio.mimsave('simulation.gif', images)

def task3():
    n_a = 100
    dt = 0.001
    t_0 = 0
    T = 1e+2 #1e+4
    sigma = 0.7

    dx = 0.01
    dy = dx
    x_min = -2.0
    x_max = 2.0
    y_min = -1.5
    y_max = 1.5

    nx, ny = (int((x_max-x_min)/dx)+1, int((y_max-y_min)/dy)+1)
    x = np.linspace(x_min, x_max, nx)
    x = np.round(x,2)
    y = np.linspace(y_min, y_max, ny)
    y = np.round(y,2)
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T
    V = (X**2-1)**2 + 3.5*Y**2

    # jacobian matrix
    grad_x_V_inner = (V[2:,:] - V[:-2,:]) / (2*dx)
    grad_y_V_inner = (V[:,2:] - V[:,:-2]) / (2*dy)

    grad_x_V = np.zeros(V.shape)
    grad_y_V = np.zeros(V.shape)
    grad_x_V[1:-1, :] = grad_x_V_inner
    grad_y_V[:, 1:-1] = grad_y_V_inner

    x_0_idx = np.random.randint(0,nx,n_a)
    y_0_idx = np.random.randint(0,ny,n_a) 

    file_nbr = 0
    for i, t in enumerate(np.linspace(t_0, T, int((T-t_0)/dt))):
        grad_V = np.array([grad_x_V[x_0_idx,y_0_idx], grad_y_V[x_0_idx,y_0_idx]])
        if i==0:
            x_t = np.array([x[x_0_idx], y[y_0_idx]])
        x_t_plus_1 = x_t - grad_V * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (2,n_a))
        x_t = x_t_plus_1

        if (np.max(x_t[0,:]) > x_max):
            idx = np.argwhere(x_t[0,:] > x_max)
            x_t[0,idx] = x_max
        if (np.min(x_t[0,:]) < x_min):
            idx = np.argwhere(x_t[0,:] < x_min)
            x_t[0,idx] = x_min
        if (np.max(x_t[1,:]) > y_max):
            idx = np.argwhere(x_t[1,:] > y_max)
            x_t[1,idx] = y_max
        if (np.min(x_t[1,:]) < y_min):
            idx = np.argwhere(x_t[1,:] < y_min)
            x_t[1,idx] = y_min
        
        x_0_idx = [int(xi) for xi in np.round((x_t[0,:]-x_min)*(nx)/(x_max-x_min)-1)]
        y_0_idx = [int(yi) for yi in np.round((x_t[1,:]-y_min)*(ny)/(y_max-y_min)-1)] 

        if i % 100 == 0 and i != 0:
            print(np.min(x_t_plus_1))
            print(np.max(x_t_plus_1))

            fig = plt.figure()
            plt.scatter(x_t[0,:], x_t[1,:])
            plt.contour(X,Y,V)
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("simulations/3_simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1


def task4():
    n_a = 300
    dt = 1.0/12 #0.001
    t_0 = 0
    T = 1e+4
    switch_t = 2500
    sigma = 0.7 # 1.0

    x_min = 8.45
    x_max = 31.55
    y_min = -10.45
    y_max = 6.5

    V_nan = np.array(pd.read_csv("landscape1.csv"))

    # values from 0 to 1 (suitible area) 
    V = 1 - V_nan
    nan_matrix = np.isnan(V)
    V[nan_matrix] = 2

    nx, ny = V.shape 
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    indices_coords_not_nan = np.argwhere(np.isnan(V_nan) == False)
    coords_not_nan = [x[indices_coords_not_nan[:,0]], y[indices_coords_not_nan[:,1]]]

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    # jacobian matrix
    grad_x_V_inner = (V[2:,:] - V[:-2,:]) / (2*dx)
    grad_y_V_inner = (V[:,2:] - V[:,:-2]) / (2*dy)

    grad_x_V = np.zeros(V.shape)
    grad_y_V = np.zeros(V.shape)
    grad_x_V[1:-1, :] = grad_x_V_inner
    grad_y_V[:, 1:-1] = grad_y_V_inner

    rand_idx = np.random.randint(0,indices_coords_not_nan.shape[0],n_a)
    x_0_idx = indices_coords_not_nan[rand_idx,0] 
    y_0_idx = indices_coords_not_nan[rand_idx,1]

    file_nbr = 0
    landscape_nbr = 1
    for i, t in enumerate(np.linspace(t_0, T, int((T-t_0)/dt))):
        if t > switch_t * landscape_nbr:
            landscape_nbr += 1
            V_nan = np.array(pd.read_csv("landscape" + str(landscape_nbr) + ".csv"))
            # values from 0 to 1 (suitible area) 
            V = 1 - V_nan
            nan_matrix = np.isnan(V)
            V[nan_matrix] = 2

            # jacobian matrix
            grad_x_V_inner = (V[2:,:] - V[:-2,:]) / (2*dx)
            grad_y_V_inner = (V[:,2:] - V[:,:-2]) / (2*dy)

            grad_x_V = np.zeros(V.shape)
            grad_y_V = np.zeros(V.shape)
            grad_x_V[1:-1, :] = grad_x_V_inner
            grad_y_V[:, 1:-1] = grad_y_V_inner

        grad_V = np.array([grad_x_V[x_0_idx,y_0_idx], grad_y_V[x_0_idx,y_0_idx]])
        if i==0:
            x_t = np.array([x[x_0_idx], y[y_0_idx]])
        x_t_plus_1 = x_t - grad_V * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (2,n_a))
        x_t = x_t_plus_1


        if (np.max(x_t[0,:]) > x_max):
            idx = np.argwhere(x_t[0,:] > x_max)
            x_t[0,idx] = x_max
        if (np.min(x_t[0,:]) < x_min):
            idx = np.argwhere(x_t[0,:] < x_min)
            x_t[0,idx] = x_min
        if (np.max(x_t[1,:]) > y_max):
            idx = np.argwhere(x_t[1,:] > y_max)
            x_t[1,idx] = y_max
        if (np.min(x_t[1,:]) < y_min):
            idx = np.argwhere(x_t[1,:] < y_min)
            x_t[1,idx] = y_min
        
        x_0_idx = [int(xi) for xi in np.round((x_t[0,:]-x_min)*(nx)/(x_max-x_min)-1)]
        y_0_idx = [int(yi) for yi in np.round((x_t[1,:]-y_min)*(ny)/(y_max-y_min)-1)] 


        for idx in range(len(x_0_idx)):
            if (nan_matrix[x_0_idx[idx], y_0_idx[idx]] == True):
                distances = np.sqrt((coords_not_nan[0] - x[x_0_idx[idx]])**2 + (coords_not_nan[1] - y[y_0_idx[idx]])**2)
                min_idx = np.argmin(distances)
                x_0_idx[idx], y_0_idx[idx] = indices_coords_not_nan[min_idx]

        if (i > 12*switch_t) & (i % (12*10) == 0): 
            print(np.min(x_t_plus_1))
            print(np.max(x_t_plus_1))

            fig = plt.figure()
            plt.scatter(x_t[0,:], x_t[1,:])
            plt.contour(X,Y,V)
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("simulations/4_simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1

    nbr_files = file_nbr
    images = []
    for i in range(nbr_files):
        filename = "simulations/4_simulation_nbr_" + str(i) + ".png"
        images.append(imageio.imread(filename))
    imageio.mimsave('4_simulation.gif', images)

def task7():
    n_a = 50
    x_max = 30
    x_min = 0
    y_max = 30
    y_min = 0
    dx = 0.1
    dy = dx

    dt = 0.01
    t_0 = 0
    T = 10

    nx, ny = (int((x_max-x_min)/dx)+1, int((y_max-y_min)/dy)+1)
    x = np.linspace(x_min, x_max, nx)
    x = np.round(x, 2)
    y = np.linspace(y_min, y_max, ny)
    y = np.round(y, 2)

    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    x_0_idx = np.random.randint(0,nx,n_a)
    y_0_idx = np.random.randint(0,ny,n_a)

    C_A = 1
    C_R = 2
    I_A = 5
    I_R = 1

    file_nbr = 0
    x_t_plus_1 = np.zeros((2,n_a))
    for i, t in enumerate(np.linspace(t_0, T, int((T-t_0)/dt))):
        if i==0:
            x_t = np.array([x[x_0_idx], y[y_0_idx]])

        positions = x_t 
        V = np.zeros(n_a)
        for idx in range(n_a):
            agent_position = positions[:,idx]
            distances = np.sqrt((positions[0,:]-agent_position[0,None])**2 + (positions[1,:]-agent_position[1,None])**2)
            V_agent = + C_A/I_A * np.exp(-distances/I_A) - C_R/I_R * np.exp(-distances/I_R) # Morse potential

            direction = agent_position[:,None] - positions
            direction = direction/np.linalg.norm(direction,axis=0)
            direction[np.isnan(direction)] = [0,0]
            change = np.sum(V_agent * direction,axis=1)
            x_t_plus_1[:,idx] = x_t[:,idx] - change * dt
            x_t[:,idx] = x_t_plus_1[:,idx]


        if (np.max(x_t[0,:]) > x_max):
            idx = np.argwhere(x_t[0,:] > x_max)
            x_t[0,idx] = x_max
        if (np.min(x_t[0,:]) < x_min):
            idx = np.argwhere(x_t[0,:] < x_min)
            x_t[0,idx] = x_min
        if (np.max(x_t[1,:]) > y_max):
            idx = np.argwhere(x_t[1,:] > y_max)
            x_t[1,idx] = y_max
        if (np.min(x_t[1,:]) < y_min):
            idx = np.argwhere(x_t[1,:] < y_min)
            x_t[1,idx] = y_min
        
        x_0_idx = [int(xi) for xi in np.round((x_t[0,:]-x_min)*(nx)/(x_max-x_min)-1)]
        y_0_idx = [int(yi) for yi in np.round((x_t[1,:]-y_min)*(ny)/(y_max-y_min)-1)] 

        if (i % (100) == 0):
            print(np.min(x_t_plus_1))
            print(np.max(x_t_plus_1))

            fig = plt.figure()
            plt.scatter(x_t[0,:], x_t[1,:])
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("simulations/7_simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1

    nbr_files = file_nbr
    images = []
    for i in range(nbr_files):
        filename = "simulations/7_simulation_nbr_" + str(i) + ".png"
        images.append(imageio.imread(filename))
    imageio.mimsave('7_simulation.gif', images)

def task81():
    n_a = 50
    x_max = 30
    x_min = 0
    y_max = 30
    y_min = 0
    dx = 0.1
    dy = dx

    dt = 0.01
    t_0 = 0
    T = 10

    sigma = 0.7

    nx, ny = (int((x_max-x_min)/dx)+1, int((y_max-y_min)/dy)+1)
    x = np.linspace(x_min, x_max, nx)
    x = np.round(x, 2)
    y = np.linspace(y_min, y_max, ny)
    y = np.round(y, 2)

    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    x_0_idx = np.random.randint(0,nx,n_a)
    y_0_idx = np.random.randint(0,ny,n_a)

    C_A = 1
    C_R = 2
    I_A = 5
    I_R = 1

    file_nbr = 0
    x_t_plus_1 = np.zeros((2,n_a))
    for i, t in enumerate(np.linspace(t_0, T, int((T-t_0)/dt))):
        if i==0:
            x_t = np.array([x[x_0_idx], y[y_0_idx]])

        positions = x_t 
        for idx in range(n_a):
            agent_position = positions[:,idx]
            distances = np.sqrt((positions[0,:]-agent_position[0,None])**2 + (positions[1,:]-agent_position[1,None])**2)
            V_agent = + C_A/I_A * np.exp(-distances/I_A) - C_R/I_R * np.exp(-distances/I_R) # Morse potential

            direction = agent_position[:,None] - positions
            direction = direction/np.linalg.norm(direction,axis=0)
            direction[np.isnan(direction)] = [0,0]
            change = np.sum(V_agent * direction,axis=1)
            x_t_plus_1[:,idx] = x_t[:,idx] - change * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, 2)
            x_t[:,idx] = x_t_plus_1[:,idx]


        if (np.max(x_t[0,:]) > x_max):
            idx = np.argwhere(x_t[0,:] > x_max)
            x_t[0,idx] = x_max
        if (np.min(x_t[0,:]) < x_min):
            idx = np.argwhere(x_t[0,:] < x_min)
            x_t[0,idx] = x_min
        if (np.max(x_t[1,:]) > y_max):
            idx = np.argwhere(x_t[1,:] > y_max)
            x_t[1,idx] = y_max
        if (np.min(x_t[1,:]) < y_min):
            idx = np.argwhere(x_t[1,:] < y_min)
            x_t[1,idx] = y_min
        
        x_0_idx = [int(xi) for xi in np.round((x_t[0,:]-x_min)*(nx)/(x_max-x_min)-1)]
        y_0_idx = [int(yi) for yi in np.round((x_t[1,:]-y_min)*(ny)/(y_max-y_min)-1)] 

        if (i % (100) == 0): 
            print(np.min(x_t_plus_1))
            print(np.max(x_t_plus_1))

            fig = plt.figure()
            plt.scatter(x_t[0,:], x_t[1,:])
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("simulations/8_simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1

    nbr_files = file_nbr
    images = []
    for i in range(nbr_files):
        filename = "simulations/8_simulation_nbr_" + str(i) + ".png"
        images.append(imageio.imread(filename))
    imageio.mimsave('8_simulation.gif', images)

def task82():
    n_a = 100 #50
    x_min = -2.0
    x_max = 2.0
    y_min = -1.5
    y_max = 1.5
    dx = 0.1
    dy = dx

    dt = 0.01
    t_0 = 0
    T = 10

    sigma = 0.7

    nx, ny = (int((x_max-x_min)/dx)+1, int((y_max-y_min)/dy)+1)
    x = np.linspace(x_min, x_max, nx)
    x = np.round(x, 2)
    y = np.linspace(y_min, y_max, ny)
    y = np.round(y, 2)

    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    x_0_idx = np.random.randint(0,nx,n_a)
    y_0_idx = np.random.randint(0,ny,n_a)


    V = (X**2-1)**2 + 3.5*Y**2

    # jacobian matrix
    grad_x_V_inner = (V[2:,:] - V[:-2,:]) / (2*dx)
    grad_y_V_inner = (V[:,2:] - V[:,:-2]) / (2*dy)

    grad_x_V = np.zeros(V.shape)
    grad_y_V = np.zeros(V.shape)
    grad_x_V[1:-1, :] = grad_x_V_inner
    grad_y_V[:, 1:-1] = grad_y_V_inner

    C_A = 0.0 #1
    I_A = 5

    C_R = 0.5 #2
    I_R = 1

    file_nbr = 0
    x_t_plus_1 = np.zeros((2,n_a))
    for i, t in enumerate(np.linspace(t_0, T, int((T-t_0)/dt))):
        if i==0:
            x_t = np.array([x[x_0_idx], y[y_0_idx]])

        positions = x_t 
        normal_distribution = np.random.normal(0, 1, (n_a,2)) #TODO
        for idx in range(n_a):
            agent_position = positions[:,idx]
            distances = np.sqrt((positions[0,:]-agent_position[0,None])**2 + (positions[1,:]-agent_position[1,None])**2)
            U_agent = + C_A/I_A * np.exp(-distances/I_A) - C_R/I_R * np.exp(-distances/I_R) # Morse potential

            direction = agent_position[:,None] - positions
            direction = direction/np.linalg.norm(direction,axis=0)
            direction[np.isnan(direction)] = np.zeros(direction[np.isnan(direction)].shape)
            change = np.sum(U_agent * direction,axis=1)
            x_t_plus_1[:,idx] = x_t[:,idx] - change * dt + sigma * np.sqrt(dt) * normal_distribution[idx]

        grad_V = np.array([grad_x_V[x_0_idx,y_0_idx], grad_y_V[x_0_idx,y_0_idx]])
        x_t_plus_1 -= grad_V * dt
        x_t = x_t_plus_1

        if (np.max(x_t[0,:]) > x_max):
            idx = np.argwhere(x_t[0,:] > x_max)
            x_t[0,idx] = x_max
        if (np.min(x_t[0,:]) < x_min):
            idx = np.argwhere(x_t[0,:] < x_min)
            x_t[0,idx] = x_min
        if (np.max(x_t[1,:]) > y_max):
            idx = np.argwhere(x_t[1,:] > y_max)
            x_t[1,idx] = y_max
        if (np.min(x_t[1,:]) < y_min):
            idx = np.argwhere(x_t[1,:] < y_min)
            x_t[1,idx] = y_min
        
        x_0_idx = [int(xi) for xi in np.round((x_t[0,:]-x_min)*(nx)/(x_max-x_min)-1)]
        y_0_idx = [int(yi) for yi in np.round((x_t[1,:]-y_min)*(ny)/(y_max-y_min)-1)] 

        if (i % (100) == 0): 
            print(np.min(x_t_plus_1))
            print(np.max(x_t_plus_1))

            fig = plt.figure()
            plt.scatter(x_t[0,:], x_t[1,:])
            plt.contour(X,Y,V)
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("simulations/82_simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1

    nbr_files = file_nbr
    images = []
    for i in range(nbr_files):
        filename = "simulations/82_simulation_nbr_" + str(i) + ".png"
        images.append(imageio.imread(filename))
    imageio.mimsave('82_simulation.gif', images)

def task9():
    n_a = 50
    x_min = -2.0
    x_max = 2.0
    y_min = -1.5
    y_max = 1.5
    dx = 0.1
    dy = dx

    dt = 0.01
    t_0 = 0

    sigma = 0.7
    gamma_12 = 0.5 
    r = 0.2 #0.5 

    nx, ny = (int((x_max-x_min)/dx)+1, int((y_max-y_min)/dy)+1)
    x = np.linspace(x_min, x_max, nx)
    x = np.round(x, 2)
    y = np.linspace(y_min, y_max, ny)
    y = np.round(y, 2)

    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    x_0_idx = np.random.randint(0,nx,n_a)
    y_0_idx = np.random.randint(0,ny,n_a)


    V = (X**2-1)**2 + 3.5*Y**2

    # jacobian matrix
    grad_x_V_inner = (V[2:,:] - V[:-2,:]) / (2*dx)
    grad_y_V_inner = (V[:,2:] - V[:,:-2]) / (2*dy)

    grad_x_V = np.zeros(V.shape)
    grad_y_V = np.zeros(V.shape)
    grad_x_V[1:-1, :] = grad_x_V_inner
    grad_y_V[:, 1:-1] = grad_y_V_inner

    C_A = 0.0 #1
    I_A = 5

    C_R = 0.5 #2
    I_R = 1

    file_nbr = 0
    initial_status = np.ones(n_a)
    x_t_plus_1 = np.zeros((3,n_a))
    tau = np.random.exponential(1)
    adaption_rate_function = np.zeros((n_a,2))
    t = t_0
    i = 0
    i_all_red = np.inf
    while i < i_all_red + 5: # five additional frames after everyone is red
        if i == 10:
            x_t[2,np.random.randint(n_a)] = 2
        if i == 0:
            x_t = np.array([x[x_0_idx], y[y_0_idx], initial_status])

        for idx in range(n_a):
            distance = np.zeros(x_t[0].shape) 
            distance[np.sqrt((x_t[0,:] - x_t[0,idx])**2 + (x_t[1,:] - x_t[1,idx])**2) < r] = 1

            delta_2 = x_t[2,:] == 2
            delta_1 = x_t[2,idx] == 1

            adaption_rate_function[idx,0] = gamma_12 * delta_1 * np.sum(distance * delta_2)
        total_adoption_rate = np.sum(adaption_rate_function)

        if total_adoption_rate * dt > tau:
            Q = adaption_rate_function / total_adoption_rate
            state_change_idx = np.argwhere(np.cumsum(Q[:,0]) > np.random.rand(1))[0] # TODO
            x_t[2,state_change_idx] = 2
            xi = np.random.normal(0,1,(n_a,2))
            scale = tau / total_adoption_rate
        else:
            xi = np.random.normal(0,1,(n_a,2))
            scale = dt
            
        positions = x_t[[0,1],:] 
        for idx in range(n_a):
            agent_position = positions[:,idx]
            distances = np.sqrt((positions[0,:]-agent_position[0,None])**2 + (positions[1,:]-agent_position[1,None])**2)
            U_agent = + C_A/I_A * np.exp(-distances/I_A) - C_R/I_R * np.exp(-distances/I_R) # Morse potential

            direction = agent_position[:,None] - positions
            direction = direction/np.linalg.norm(direction,axis=0)
            direction[np.isnan(direction)] = np.zeros(direction[np.isnan(direction)].shape)
            change = np.sum(U_agent * direction,axis=1)
            x_t_plus_1[[0,1],idx] = x_t[[0,1],idx] - change * scale + sigma * np.sqrt(scale) * xi[idx]

        grad_V = np.array([grad_x_V[x_0_idx,y_0_idx], grad_y_V[x_0_idx,y_0_idx]])
        x_t_plus_1[[0,1],:] -= grad_V * scale
        x_t[[0,1],:] = x_t_plus_1[[0,1],:]

        # agents leaving domain because of noise
        if (np.max(x_t[0,:]) > x_max):
            idx = np.argwhere(x_t[0,:] > x_max)
            x_t[0,idx] = x_max
        if (np.min(x_t[0,:]) < x_min):
            idx = np.argwhere(x_t[0,:] < x_min)
            x_t[0,idx] = x_min
        if (np.max(x_t[1,:]) > y_max):
            idx = np.argwhere(x_t[1,:] > y_max)
            x_t[1,idx] = y_max
        if (np.min(x_t[1,:]) < y_min):
            idx = np.argwhere(x_t[1,:] < y_min)
            x_t[1,idx] = y_min
        
        x_0_idx = [int(xi) for xi in np.round((x_t[0,:]-x_min)*(nx)/(x_max-x_min)-1)]
        y_0_idx = [int(yi) for yi in np.round((x_t[1,:]-y_min)*(ny)/(y_max-y_min)-1)] 

        if (i >= 10): 
            print(np.min(x_t_plus_1[[0,1],:]))
            print(np.max(x_t_plus_1[[0,1],:]))

            fig = plt.figure()
            blue_agents = x_t[:,x_t[2,:] == 1]
            red_agents = x_t[:,x_t[2,:] == 2]
            plt.scatter(blue_agents[0,:], blue_agents[1,:], color="blue")
            plt.scatter(red_agents[0,:], red_agents[1,:], color="red")
            plt.contour(X,Y,V)
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("simulations/9_simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1
        
        if total_adoption_rate * dt > tau:
            t += tau/total_adoption_rate
            tau = np.random.exponential(1)
        else:
            t += dt
            tau -= total_adoption_rate * dt
        
        if np.all(x_t[2,:] == 2) & (i_all_red == np.inf):
            i_all_red = i
        i += 1

    nbr_files = file_nbr
    images = []
    for i in range(nbr_files):
        filename = "simulations/9_simulation_nbr_" + str(i) + ".png"
        images.append(imageio.imread(filename))
    imageio.mimsave('9_simulation.gif', images)

def task10():
    n_a = 300
    dt = 1.0/12 
    t_0 = 0
    T = 1e+4
    switch_t = 2500

    x_min = 8.45
    x_max = 31.55
    y_min = -10.45
    y_max = 6.5

    sigma = 0.7
    gamma_12 = 0.002 # interaction rate
    r = 1.0 #0.5 # interaction radius

    C_A = 0.0 #1
    I_A = 5

    C_R = 0.5 #2
    I_R = 1

    V_nan = np.array(pd.read_csv("landscape1.csv"))

    # values from 0 to 1 (suitible area) 
    V = 1 - V_nan
    nan_matrix = np.isnan(V)
    V[nan_matrix] = 2

    nx, ny = V.shape
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    indices_coords_not_nan = np.argwhere(np.isnan(V_nan) == False)
    coords_not_nan = [x[indices_coords_not_nan[:,0]], y[indices_coords_not_nan[:,1]]]

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    # jacobian matrix
    grad_x_V_inner = (V[2:,:] - V[:-2,:]) / (2*dx)
    grad_y_V_inner = (V[:,2:] - V[:,:-2]) / (2*dy)

    grad_x_V = np.zeros(V.shape)
    grad_y_V = np.zeros(V.shape)
    grad_x_V[1:-1, :] = grad_x_V_inner
    grad_y_V[:, 1:-1] = grad_y_V_inner

    rand_idx = np.random.randint(0,indices_coords_not_nan.shape[0],n_a)
    x_0_idx = indices_coords_not_nan[rand_idx,0]
    y_0_idx = indices_coords_not_nan[rand_idx,1]

    landscape_nbr = 1
    file_nbr = 0
    initial_status = np.ones(n_a)
    x_t_plus_1 = np.zeros((3,n_a))
    tau = np.random.exponential(1)
    adaption_rate_function = np.zeros((n_a,2))
    t = t_0
    i = 0
    i_all_red = np.inf
    delay_t = 4e+3
    delay = True
    while i < i_all_red + 5: # five additional frames after everyone is red
        if t > switch_t * landscape_nbr:
            landscape_nbr += 1
            V_nan = np.array(pd.read_csv("landscape" + str(landscape_nbr) + ".csv"))
            # values from 0 to 1 (suitible area) 
            V = 1 - V_nan
            nan_matrix = np.isnan(V)
            V[nan_matrix] = 2

            # jacobian matrix
            grad_x_V_inner = (V[2:,:] - V[:-2,:]) / (2*dx)
            grad_y_V_inner = (V[:,2:] - V[:,:-2]) / (2*dy)

            grad_x_V = np.zeros(V.shape)
            grad_y_V = np.zeros(V.shape)
            grad_x_V[1:-1, :] = grad_x_V_inner
            grad_y_V[:, 1:-1] = grad_y_V_inner

        if (delay) & (t >= delay_t):
            x_t[2,np.random.randint(n_a)] = 2
            delay = False
        if i == 0:
            x_t = np.array([x[x_0_idx], y[y_0_idx], initial_status])

        for idx in range(n_a):
            distance = np.zeros(x_t[0].shape) 
            distance[np.sqrt((x_t[0,:] - x_t[0,idx])**2 + (x_t[1,:] - x_t[1,idx])**2) < r] = 1

            delta_2 = x_t[2,:] == 2
            delta_1 = x_t[2,idx] == 1

            adaption_rate_function[idx,0] = gamma_12 * delta_1 * np.sum(distance * delta_2)
        total_adoption_rate = np.sum(adaption_rate_function)

        if total_adoption_rate * dt > tau:
            Q = adaption_rate_function / total_adoption_rate
            state_change_idx = np.argwhere(np.cumsum(Q[:,0]) > np.random.rand(1))[0] # TODO
            x_t[2,state_change_idx] = 2
            xi = np.random.normal(0,1,(n_a,2))
            scale = tau / total_adoption_rate
        else:
            xi = np.random.normal(0,1,(n_a,2))
            scale = dt
            
        positions = x_t[[0,1]] 
        for idx in range(n_a):
            agent_position = positions[:,idx]
            distances = np.sqrt((positions[0,:]-agent_position[0,None])**2 + (positions[1,:]-agent_position[1,None])**2)
            U_agent = + C_A/I_A * np.exp(-distances/I_A) - C_R/I_R * np.exp(-distances/I_R) # Morse potential

            direction = agent_position[[0,1],None] - positions
            direction = direction/np.linalg.norm(direction,axis=0)
            direction[np.isnan(direction)] = np.zeros(direction[np.isnan(direction)].shape)
            change = np.sum(U_agent * direction,axis=1)
            x_t_plus_1[[0,1],idx] = x_t[[0,1],idx] - 0.1**2 * change * scale + sigma * np.sqrt(scale) * xi[idx]

        grad_V = np.array([grad_x_V[x_0_idx,y_0_idx], grad_y_V[x_0_idx,y_0_idx]])
        x_t_plus_1[[0,1],:] -= grad_V * scale
        x_t[[0,1],:] = x_t_plus_1[[0,1],:]

        if (np.max(x_t[0,:]) > x_max):
            idx = np.argwhere(x_t[0,:] > x_max)
            x_t[0,idx] = x_max
        if (np.min(x_t[0,:]) < x_min):
            idx = np.argwhere(x_t[0,:] < x_min)
            x_t[0,idx] = x_min
        if (np.max(x_t[1,:]) > y_max):
            idx = np.argwhere(x_t[1,:] > y_max)
            x_t[1,idx] = y_max
        if (np.min(x_t[1,:]) < y_min):
            idx = np.argwhere(x_t[1,:] < y_min)
            x_t[1,idx] = y_min
        
        x_0_idx = [int(xi) for xi in np.round((x_t[0,:]-x_min)*(nx)/(x_max-x_min)-1)]
        y_0_idx = [int(yi) for yi in np.round((x_t[1,:]-y_min)*(ny)/(y_max-y_min)-1)] 

        for idx in range(len(x_0_idx)):
            if (nan_matrix[x_0_idx[idx], y_0_idx[idx]] == True):
                distances = np.sqrt((coords_not_nan[0] - x[x_0_idx[idx]])**2 + (coords_not_nan[1] - y[y_0_idx[idx]])**2)
                min_idx = np.argmin(distances)
                x_0_idx[idx], y_0_idx[idx] = indices_coords_not_nan[min_idx]

        if (i % (12*10) == 0) & (t >= delay_t): 
            print(np.min(x_t_plus_1[[0,1],:]))
            print(np.max(x_t_plus_1[[0,1],:]))

            fig = plt.figure()
            blue_agents = x_t[:,x_t[2,:] == 1]
            red_agents = x_t[:,x_t[2,:] == 2]
            plt.scatter(blue_agents[0,:], blue_agents[1,:], color="blue")
            plt.scatter(red_agents[0,:], red_agents[1,:], color="red")
            plt.contour(X,Y,V)
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("simulations/10_simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1
        
        if total_adoption_rate * dt > tau:
            t += tau/total_adoption_rate
            tau = np.random.exponential(1)
        else:
            t += dt
            tau -= total_adoption_rate * dt
        
        if np.all(x_t[2,:] == 2) & (i_all_red == np.inf):
            i_all_red = i
        i += 1

    nbr_files = file_nbr
    images = []
    for i in range(nbr_files):
        filename = "simulations/10_simulation_nbr_" + str(i) + ".png"
        images.append(imageio.imread(filename))
    imageio.mimsave('10_simulation.gif', images)


if __name__ == "__main__":
    #vectorized()

    # task3()

    # task4()

    # task7()

    # task81()

    # task82()

    # task9()

    #task10()