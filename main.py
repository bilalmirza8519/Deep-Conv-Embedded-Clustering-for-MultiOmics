np.random.seed(5)# 
from tensorflow import set_random_seed
set_random_seed(5)


dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=6)
#plot_model(dcec.model, to_file='results/temp' + '/dcec_model.png', show_shapes=True)
dcec.model.summary()

    # begin clustering.
optimizer = 'adam'

dcec.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)
#dcec.compile(loss='kld', optimizer=optimizer)
dcec.fit(x, tol=0.001, maxiter=2e4,
            update_interval=140,            
            cae_weights=None,batch_size=256)  
y_pred = dcec.y_pred
