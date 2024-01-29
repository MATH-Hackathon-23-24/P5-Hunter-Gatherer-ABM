import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd

def taskfinal():
    n_a = 300 # number of agents

    # domain
    x_min = 8.45
    x_max = 31.55
    y_min = -10.45
    y_max = 6.5

    sigma = 0.7
    gamma_first_order_events = 0.001 # rate for random state change
    gamma_second_order_events = 0.01 # interaction rate
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
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    X, Y = np.meshgrid(x, y)
    X = X.T
    Y = Y.T

    indices_coords_not_nan = np.argwhere(np.isnan(V_nan) == False)
    coords_not_nan = [x[indices_coords_not_nan[:,0]], y[indices_coords_not_nan[:,1]]]
 
    # jacobian matrix
    grad_x_V_inner = (V[2:,:] - V[:-2,:]) / (2*dx)
    grad_y_V_inner = (V[:,2:] - V[:,:-2]) / (2*dy)

    grad_x_V = np.zeros(V.shape)
    grad_y_V = np.zeros(V.shape)
    grad_x_V[1:-1, :] = grad_x_V_inner
    grad_y_V[:, 1:-1] = grad_y_V_inner

    # initial indices of position of agents
    rand_idx = np.random.randint(0,indices_coords_not_nan.shape[0],n_a)
    x_0_idx = indices_coords_not_nan[rand_idx,0]
    y_0_idx = indices_coords_not_nan[rand_idx,1]

    landscape_nbr = 1
    file_nbr = 0
    nbr_features = 3
    traits = 10
    initial_status = np.random.randint(1,traits+1,(nbr_features,n_a))
    x_t_plus_1 = np.zeros((2+nbr_features,n_a))
    tau = np.random.exponential(1)
    adaption_rate_function_1st = gamma_first_order_events * np.ones((n_a,traits))
    adaption_rate_function_2nd = np.zeros((n_a,nbr_features,traits,traits))
    
    i = 0
    delay_t = 4e+3
    delay = True
    dt = 1.0/12 # 12 months per year (t=1)
    t_0 = 0
    T = 14e+3
    switch_t = 2500 #switch landscape
    t = t_0
    while t < T:
        if t > switch_t * landscape_nbr:
            landscape_nbr += 1
            V_nan = np.array(pd.read_csv("landscape" + str(landscape_nbr) + ".csv"))
            # values from 0 to 1 (suitible area) -> change so that 0 is suitable
            V = 1 - V_nan
            nan_matrix = np.isnan(V)
            V[nan_matrix] = 2

            grad_x_V = (V[2:,:]-V[:-2,:])/(2*dx)  #(V(x+dx,y) - V(x-dx,y))/(2*dx) 
            grad_y_V = (V[:,2:]-V[:,:-2])/(2*dy)  #(V(x,y+dy) - V(x,y-dy))/(2*dy) 

            grad_x_V = np.concatenate([((V[1,:]-V[0,:])/(2*dx))[None,:], grad_x_V])
            grad_x_V = np.concatenate([grad_x_V, ((V[-1,:]-V[-2,:])/(2*dx))[None,:]])

            grad_y_V = np.concatenate([((V[:,1]-V[:,0])/(2*dy))[:,None], grad_y_V], axis=1)
            grad_y_V = np.concatenate([grad_y_V,((V[:,-1]-V[:,-2])/(2*dy))[:,None]], axis=1)

            grad_x_V[[0,-1],:] = 0
            grad_y_V[[0,-1],:] = 0
            grad_x_V[:,[0,-1]] = 0
            grad_y_V[:,[0,-1]] = 0

        if i == 0:
            x_t = np.concatenate([x[x_0_idx][:,None].T, y[y_0_idx][:,None].T, initial_status], axis=0)

        for idx in range(n_a):
            distance = np.zeros(x_t[0].shape)
            distance[np.sqrt((x_t[0,:] - x_t[0,idx])**2 + (x_t[1,:] - x_t[1,idx])**2) < r] = 1
            
            for feature in range(nbr_features):
                delta_agent_1 = x_t[2+feature,idx] == np.arange(1,traits+1)
                delta_agents = np.zeros((traits,n_a))
                for trait in range(1,traits+1):
                    delta_agents[trait-1] = x_t[2+feature,:] == trait

                adaption_rate_function_2nd[idx,feature,:,:] = gamma_second_order_events * delta_agent_1[:,None] * np.sum(distance[:,None] * delta_agents.T, axis=0)[None,:]  # 300 x 10 before sum, 10 after sum

        # no transition to the same trait
        for feature in range(nbr_features):
            for trait in range(traits):
                adaption_rate_function_2nd[:,feature,trait,trait] = 0

        total_adoption_rate = np.sum(adaption_rate_function_2nd) + np.sum(adaption_rate_function_1st)
        if total_adoption_rate * dt > tau:
            if np.random.rand(1) > np.sum(adaption_rate_function_2nd) / total_adoption_rate:
                agent_change_idx = np.random.randint(0,n_a)             # random agent selection
                feature_change_idx = np.random.randint(0,nbr_features)  # random feature selection
                trait_change_to_idx = np.random.randint(0,traits)       # random trait selection
            else:
                # agent selection
                Q1 = adaption_rate_function_2nd.sum(axis=(1, 2, 3)) 
                Q1 /= Q1.sum()
                agent_change_idx = np.argwhere(np.cumsum(Q1) > np.random.rand(1))[0]

                # feature selection
                Q2 = np.squeeze(adaption_rate_function_2nd[agent_change_idx]).sum(axis=(1, 2))
                Q2 /= Q2.sum()
                feature_change_idx = np.argwhere(np.cumsum(Q2) > np.random.rand(1))[0]

                # trait selection
                Q3 = np.squeeze(adaption_rate_function_2nd[agent_change_idx,feature_change_idx]).sum(axis=1)
                Q3 /= Q3.sum()
                trait_change_from_idx = np.argwhere(np.cumsum(Q3) > np.random.rand(1))[0]

                Q4 = np.squeeze(adaption_rate_function_2nd[agent_change_idx,feature_change_idx,trait_change_from_idx])
                Q4 /= Q4.sum()
                trait_change_to_idx = np.argwhere(np.cumsum(Q4) > np.random.rand(1))[0]

            # changing trait
            x_t[2+feature_change_idx,agent_change_idx] = trait_change_to_idx + 1

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
            x_t_plus_1[[0,1],idx] = x_t[[0,1],idx] - 0.1 * change * scale + sigma * np.sqrt(scale) * xi[idx]

        grad_V = np.array([grad_x_V[x_0_idx,y_0_idx], grad_y_V[x_0_idx,y_0_idx]])
        x_t_plus_1[[0,1],:] -= grad_V * scale
        x_t[[0,1],:] = x_t_plus_1[[0,1],:] # new positions

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
        
        # getting indices of positions
        x_0_idx = [int(xi) for xi in np.round((x_t[0,:]-x_min)*(nx)/(x_max-x_min)-1)]
        y_0_idx = [int(yi) for yi in np.round((x_t[1,:]-y_min)*(ny)/(y_max-y_min)-1)] 

        # check for agents wandering off into the ocean aka domain with NaN values
        for idx in range(len(x_0_idx)):
            if (nan_matrix[x_0_idx[idx], y_0_idx[idx]] == True):
                distances = np.sqrt((coords_not_nan[0] - x[x_0_idx[idx]])**2 + (coords_not_nan[1] - y[y_0_idx[idx]])**2)
                min_idx = np.argmin(distances)
                x_0_idx[idx], y_0_idx[idx] = indices_coords_not_nan[min_idx]

        # start plotting after some waiting time
        if (i % (12*10) == 0) & (t >= delay_t): 
            print(np.min(x_t_plus_1[[0,1],:]))
            print(np.max(x_t_plus_1[[0,1],:]))

            fig = plt.figure()
            plt.scatter(x_t[0,:], x_t[1,:], color = np.array((x_t[2,:]/traits, x_t[3,:]/traits, x_t[4,:]/traits)).T, #RGB color
                        edgecolors='black')
            plt.contour(X,Y,V, cmap="Greys")
            plt.axis([x_min, x_max, y_min, y_max])

            plt.savefig("final_simulations/final_simulation_nbr_" + str(file_nbr) + ".png")
            plt.close(fig)
            file_nbr += 1
        
        if total_adoption_rate * dt > tau:
            t += tau/total_adoption_rate
            tau = np.random.exponential(1)
        else:
            t += dt
            tau -= total_adoption_rate * dt
        i += 1

    nbr_files = file_nbr
    images = []
    for i in range(nbr_files):
        filename = "final_simulations/final_simulation_nbr_" + str(i) + ".png"
        images.append(imageio.imread(filename))
    imageio.mimsave('final_simulation.gif', images)


if __name__ == "__main__":
    taskfinal()