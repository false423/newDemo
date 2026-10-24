import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, NMF

def pca_img(x, n_components):
    Data = x
    m, n, l = x.shape
    # extract the first principal component
    x = np.reshape(Data, (m * n, l))
    pca = PCA(n_components, copy=True, whiten=False)
    x = pca.fit_transform(x)
    _, l = x.shape
    x = np.reshape(x, (m, n, l))
    #print(x.shape)
    return x

def lda_img(x,x_width,x_length,x_bands, n_components):
    Data = x
    m, n, l = x_width, x_length, x_bands
    # extract the first principal component
    x = np.reshape(Data, (m * n, l))

    nmf = NMF(n_components=3, init='random', random_state=0)
    X_lle = nmf.fit_transform(x)

    _, l = X_lle.shape
    x = np.reshape(X_lle, (m, n, l))
    print(x.shape)
    return x

def lda_img1(x,x_width,x_length,x_bands, n_components):
    Data = x
    m, n, l = x_width, x_length, x_bands
    # extract the first principal component
    x = np.reshape(Data, (m * n, l))

    tsne = TruncatedSVD(n_components=3, random_state=0)
    X_tsne = tsne.fit_transform(x)

    _, l = X_tsne.shape
    x = np.reshape(X_tsne, (m, n, l))
    print(x.shape)
    
    return x


