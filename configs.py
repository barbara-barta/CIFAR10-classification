learning_rate = 0.003
num_epochs = 150
batch_size = 4
patience = 20

gamma = 0.5
step_size = 10

num_workers_local = 0
num_workers_cloud = 6

mean = [0.4914, 0.4822, 0.4465]
std  = [0.2470, 0.2435, 0.2616]

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

model_name =  f"SixLayers_pool2_da_sch_bn_skip_{gamma}_{step_size}_{learning_rate}_{num_epochs}_{patience}"
