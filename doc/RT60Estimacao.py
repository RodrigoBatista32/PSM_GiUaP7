# PSM - Aula Prática 23-10-2026
# Estimação do Tempo de Reverberação de uma sala
# Utilização do método de Integração reversa - Backwards Integration


import numpy as np
from scipy.io import wavfile
from scipy.stats import linregress
import matplotlib.pyplot as plt

caminho_ficheiro = 'AulaPratica_23_10_26.wav'

# 1. Ler o ficheiro WAV
fs, dados = wavfile.read(caminho_ficheiro)

# Se o áudio for estéreo (mais de uma coluna), selecionamos apenas o primeiro canal
if len(dados.shape) > 1:
    dados = dados[:, 0]

dados = dados.astype('float32')

plt.figure()
plt.plot(dados)
plt.show()  

hIR = dados[323000:600000] # por inspeção do gráfico, retira-se uma Resposta ao Impulso

hIR2 = hIR**2
hIR2 = hIR2[26900: 53400]  # por inspeção do gráfico, retira-se a parte da Resposta ao Impulso
                            # que ainda não entrou no ruido de fundo 

hIR2Energy = 10*np.log10(hIR2)

BackwardIntegration = np.cumsum(hIR2[::-1])[::-1] / fs
BackwardIntegration = 10*np.log10(BackwardIntegration)


# plt.figure()
# plt.plot(hIR2)
# plt.show()
# plt.figure()
# plt.plot(BackwardIntegration)
# plt.show()

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(hIR)
plt.subplot(3, 1, 2)
plt.plot(hIR2)
plt.subplot(3, 1, 3)
plt.plot(hIR2Energy)
plt.plot(BackwardIntegration, 'r')
plt.show()  

BackwardIntegration = BackwardIntegration[0: 20000] # Usa-se a porção da curva que melhor se aproxima de uma reta

# 2. Criar o eixo temporal (x) baseado no número de amostras
n_amostras = len(BackwardIntegration)
x = np.arange(n_amostras) / fs

# 3. Realizar a regressão linear
# linregress retorna: declive, interceptação, coeficiente_correlacao, p_value, stderr
slope, intercept, r_value, p_value, std_err = linregress(x, BackwardIntegration)

RT60 = np.abs(60 / slope)

print(f"Ficheiro: {caminho_ficheiro}")
print(f"Declive (m): {slope}")
print(f"Ordenada na origem (b): {intercept}")
print(f"Coeficiente de correlação (r): {r_value}")
print(f"Reta ajustada: y = {slope:.6f}x + {intercept:.6f}")
print(f"RT60 (s): {np.round(RT60, 2)}")
