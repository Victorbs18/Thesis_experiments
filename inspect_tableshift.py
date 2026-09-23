from tableshift import get_dataset

dset = get_dataset('diabetes_readmission')

X_tr,  y_tr,  g_tr,  _ = dset.get_pandas('train')
X_val, y_val, g_val, _ = dset.get_pandas('validation')
X_id,  y_id,  g_id,  _ = dset.get_pandas('id_test')
X_ood, y_ood, g_ood, _ = dset.get_pandas('ood_test')

print('Train shape:', X_tr.shape)
print('Val shape:  ', X_val.shape)
print('ID test:    ', X_id.shape)
print('OOD test:   ', X_ood.shape)
print()
print('Train environments:', sorted(g_tr.unique().tolist()))
print('Val environments:  ', sorted(g_val.unique().tolist()))
print('ID test envs:      ', sorted(g_id.unique().tolist()))
print('OOD test envs:     ', sorted(g_ood.unique().tolist()))
print()
print('n_classes:', len(y_tr.unique()))
print('in_dim:   ', X_tr.shape[1])
print('Class balance train:', y_tr.value_counts().to_dict())
print('Class balance OOD:  ', y_ood.value_counts().to_dict())