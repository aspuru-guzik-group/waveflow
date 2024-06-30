from waveflow import vqmc

box_length = 10
log_every = 250
batch_size  =32
num_epochs = 1000 # it might over shoot
n_knots = 11
n_layer = 3
spline_degree = 4
trainer = vqmc.ModelTrainer(num_epochs=num_epochs, box_length=box_length,
                            batch_size=batch_size, log_every=log_every)
trainer.num_knots = n_knots
trainer.n_flow_layer = n_layer 
trainer.spline_degree = spline_degree
trainer.start_training()