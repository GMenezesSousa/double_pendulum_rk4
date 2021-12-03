### Importando as Bibliotecas ###

import numpy as np
import matplotlib.pyplot as plt

my_form = "{0:.2f}"

def rk4(a, b, h, y0, y):
    '''
    Esta rotina resolve numericamente um sistema de equações diferenciais de
    primeira ordem pelo método de Runge-Kutta de quarta ordem:

                        Y = F(t, y)

    onde Y, F e y são vetores com n elementos. O vetor y é aquele que contém as
    equações diferenciais do problema em questão, podendo ser um sistema de n
    equações. Obs: Este vetor y deve ser uma função que retorna um array contendo
    as equações diferenciais do problema.

    Parâmetros:
    (a,b)         -> Intervalo de integração de interesse
    h             -> Tamanho dos passos
    y0            -> Vetor contendo as condições iniciais
    y             -> Vetor contendo as equações diferenciais
    '''
    
    # Primeiramente precisamos construir nossos arrays iniciais

    t = np.arange(a, b, h)           # Inicializando o array tempo
    num = len(y0)                    # Número de equações diferenciais
    tam = len(t)                     # O comprimento do Array

    solution = np.zeros((tam, num))  # Inicializando o vetor solução

    solution[0,:] = y0               # Definindo as condições iniciais
    for i in range(0, tam-1, 1):
        # Calculemos as inclinações para cada função
        k1 = y(t[i], solution[i,:])                 # Primeira inclinação
        k2 = y(t[i] + h/2, solution[i,:] + k1*h/2)  # Segunda inclinação
        k3 = y(t[i] + h/2, solution[i,:] + k2*h/2)  # Terceira inclinação
        k4 = y(t[i] + h, solution[i,:] + k3*h)      # Quarta inclinação

        # Calculemos as estimativa da iteração posterior

        solution[i+1,:] = solution[i,:] + (k1 + 2*k2 + 2*k3 + k4)*(h/6)

    return t, solution

def equation(t, y0):
    '''
    Esta função calcula as equações de movimento para o tempo atual, t, e uma
    iteração anterior y0. Obs: Esta função tem que ser específica para o seu
    problema, neste caso, trataremos do movimento estelar sujeito ao potencial
    galáctico e sua perturbação devido aos braços espirais.

    Parâmetros:

    t    -> Tempo
    y0   -> Array contendo as equações calculadas em um tempo anterior.
            
    Caso do pêndulo duplo: y0 = (t, theta1_0, theta2_0, p1_0, p2_0)
    '''
    theta1_0 = y0[0]
    theta2_0 = y0[1]
    p1_0 = y0[2]
    p2_0 = y0[3]
    
    M = m1 + m2
    a = l1*l2*(m1 + m2*(np.sin(theta1_0 - theta2_0))**2)
    
    C1 = p1_0*p2_0*np.sin(theta1_0 - theta2_0)/a
    C2 = (l2**2*m2*p1_0**2 + l1**2*M*p2_0**2 - l1*l2*m2*p1_0*p2_0*np.cos(theta1_0 - theta2_0))*np.sin(2*(theta1_0 - theta2_0))/(2*a**2)
    
    dtheta1 = (l2*p1_0 - l1*p2_0*np.cos(theta1_0 - theta2_0))/(l1*a)
    dtheta2 = (l1*M*p2_0 - l2*m2*p1_0*np.cos(theta1_0 - theta2_0))/(l2*m2*a)
    
    dp1 = (-1)*M*g*l1*np.sin(theta1_0) - C1 + C2
    dp2 = (-1)*m2*g*l2*np.sin(theta2_0) + C1 - C2
    
    
    return np.array([dtheta1, dtheta2, dp1, dp2])

def equation2(t, y0):
    '''
    Esta função calcula as equações de movimento para o tempo atual, t, e uma
    iteração anterior y0. Obs: Esta função tem que ser específica para o seu
    problema, neste caso, trataremos do movimento estelar sujeito ao potencial
    galáctico e sua perturbação devido aos braços espirais.

    Parâmetros:

    t    -> Tempo
    y0   -> Array contendo as equações calculadas em um tempo anterior.
            
    Caso do pêndulo duplo: y0 = (t, theta1_0, theta2_0, w1_0, w2_0)
    '''
    theta1_0 = y0[0]
    theta2_0 = y0[1]
    w1_0 = y0[2]
    w2_0 = y0[3]
    
    b = 2*m1 + m2*(1 - np.cos(2*(theta1_0 - theta2_0)))
    
    dtheta1 = w1_0
    dtheta2 = w2_0
    
    c1 = (-1)*g*(2*m1 + m2)*np.sin(theta1_0)
    c2 = m2*g*np.sin(theta1_0 - 2*theta2_0)
    c3 = 2*np.sin(theta1_0 - theta2_0)*m2*(w2_0**2*l2 + w1_0**2*l1*np.cos(theta1_0 - theta2_0))
    
    dw1 = (c1 - c2 - c3)/(l1*b)
    
    c4 = w1_0**2*l1*(m1 + m2)
    c5 = g*(m1 + m2)*np.cos(theta1_0)
    c6 = w2_0**2*l2*m2*np.cos(theta1_0 - theta2_0)
    
    dw2 = 2*np.sin(theta1_0 - theta2_0)*(c4 + c5 + c6)/(l2*b)    
    
    return np.array([dtheta1, dtheta2, dw1, dw2])

def simula_pendulo(x0, y0, tf, dt):
    # y0 é o array com as condições iniciais, [theta1_0, theta2_0, omega1_0, omega2_0].
    # x0 é o array com as caracteristicas do problema, [l1, l2, m1, m2, g]
    l1 = x0[0]
    l2 = x0[1]
    m1 = x0[2]
    m2 = x0[3]
    g = x0[4]
    
    theta1_0 = y0[0]
    theta2_0 = y0[1]
    w1_0 = y0[2]
    w2_0 = y0[3]
    
    
    t, sol = rk4(0, tf, dt, y0, equation2)
    plt.plot(t, sol[:, 1])
    plt.show()
    
    t1 = my_form.format(theta1_0)
    t2 = my_form.format(theta2_0)
    w1 = my_form.format(w1_0)
    w2 = my_form.format(w2_0)
    
    titulo = r'$\theta_1$ = ' + t1
    titulo += ', '
    titulo += r'$\theta_2$ = '
    titulo += t2
    titulo += ', '
    titulo += r'$\omega_1$ = '
    titulo += w1
    titulo += ', '
    titulo += r'$\omega_2$ = '
    titulo += w2
    
    plt.plot(sol[:, 0], sol[:, 2])
    plt.title(titulo)
    plt.xlabel(r'$\theta_1$ (rad)')
    plt.ylabel(r'$\omega_1$')
    plt.show()
    
    plt.plot(sol[:, 1], sol[:, 3])
    plt.title(titulo)
    plt.xlabel(r'$\theta_2$ (rad)')
    plt.ylabel(r'$\omega_2$')
    plt.show()

    p1_0 = (m1 + m2)*l1**2*w1_0 + m2*l1*l2*w2_0*np.cos(theta2_0 - theta1_0)
    p2_0 = m2*l2**2*w2_0 + m2*l1*l2*w1_0*np.cos(theta2_0 - theta1_0)
    
    y0 = np.array([theta1_0, theta2_0, p1_0, p2_0])
    
    t, sol = rk4(0, tf, dt, y0, equation)
    plt.plot(t, sol[:, 1])
    plt.show()
    
    t1 = my_form.format(theta1_0)
    t2 = my_form.format(theta2_0)
    p1 = my_form.format(p1_0)
    p2 = my_form.format(p2_0)
    
    titulo = r'$\theta_1$ = ' + t1
    titulo += ', '
    titulo += r'$\theta_2$ = '
    titulo += t2
    titulo += ', '
    titulo += r'$p_1$ = '
    titulo += p1
    titulo += ', '
    titulo += r'$p_2$ = '
    titulo += p2
    
    plt.plot(sol[:, 0], sol[:, 2])
    plt.title(titulo)
    plt.xlabel(r'$\theta_1$ (rad)')
    plt.ylabel(r'p_$\theta$')
    plt.show()
    
    plt.plot(sol[:, 1], sol[:, 3])
    plt.title(titulo)
    plt.xlabel(r'$\theta_2$ (rad)')
    plt.ylabel(r'p_$\theta$')
    plt.show()

    return None

def compara_pw(x0, y0, tf, dt):
    # y0 é o array com as condições iniciais, [theta1_0, theta2_0, omega1_0, omega2_0].
    # x0 é o array com as caracteristicas do problema, [l1, l2, m1, m2, g]
    l1 = x0[0]
    l2 = x0[1]
    m1 = x0[2]
    m2 = x0[3]
    g = x0[4]
    
    theta1_0 = y0[0]
    theta2_0 = y0[1]
    w1_0 = y0[2]
    w2_0 = y0[3]
    
    
    t, sol = rk4(0, tf, dt, y0, equation2)
    lista_theta1 = sol[:, 0]
    lista_theta2 = sol[:, 1]
    lista_w1 = sol[:, 2]
    lista_w2 = sol[:, 3]
    lista_p1 = (m1 + m2)*l1**2*lista_w1 + m2*l1*l2*lista_w2*np.cos(lista_theta1 - lista_theta2)
    lista_p2 = m2*l2**2*lista_w2 + m2*l1*l2*lista_w1*np.cos(lista_theta1 - lista_theta2)

    p1_0 = (m1 + m2)*l1**2*w1_0 + m2*l1*l2*w2_0*np.cos(theta2_0 - theta1_0)
    p2_0 = m2*l2**2*w2_0 + m2*l1*l2*w1_0*np.cos(theta2_0 - theta1_0)
    
    y0 = np.array([theta1_0, theta2_0, p1_0, p2_0])
    
    t, sol = rk4(0, tf, dt, y0, equation)
    
    plt.plot(sol[:, 2], lista_p1)
    plt.plot(lista_p1, lista_p1, '-r')
    plt.grid()
    plt.title('p1 x p1')
    plt.xlabel('p1 (equations 1)')
    plt.ylabel('p1 (calculado de w1 e w2)')
    plt.show()
    
    plt.plot(sol[:, 3], lista_p2)
    plt.plot(lista_p2, lista_p2, '-r')
    plt.grid()
    plt.title('p2 x p2')
    plt.xlabel('p2 (equations 1)')
    plt.ylabel('p2 (calculado de w1 e w2)')
    plt.show()

    return None

if __name__ == '__main__':
    l1 = 1
    l2 = 1
    m1 = 2
    m2 = 2
    g = 9.81
    x0 = np.array([l1, l2, m1, m2, g])
    
    theta1_0 = 60*np.pi/180
    theta2_0 = 60*np.pi/180
    w1_0 = 0
    w2_0 = 0
    y0 = np.array([theta1_0, theta2_0, w1_0, w2_0])
    
    #simula_pendulo(x0, y0, 50, 0.01)
    compara_pw(x0, y0, 10, 0.01)
    
    