# Stepper
def stepper(func, dt, t, X_cur):
    # Euler
    # dX_cur = func(t, X_cur)
    # X_new = X_cur + dt*dX_cur

    # Runge Kutta de Quarta Ordem
    tbarra = t + dt*0.5
    dX_1 = func(t, X_cur)
    k1 = dt*dX_1
    
    X_barra = X_cur + k1*0.5
    dX_2 = func(tbarra, X_barra)
    k2 = dt*dX_2

    X_barra = X_cur + k2*0.5
    dX_3 = func(tbarra, X_barra)
    k3 = dt*dX_3

    X_barra = X_cur + k3
    tbarra = t + dt
    dX_4 = func(tbarra, X_barra)
    k4 = dt*dX_4

    X_new  =  X_cur  +  ( k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    
    return X_new