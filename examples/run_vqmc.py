from waveflow import vqmc

num_epochs = 80000 # it might over shoot
trainer = vqmc.ModelTrainer(num_epochs=num_epochs)
trainer.start_training()