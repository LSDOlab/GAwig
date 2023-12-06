import numpy as np

def cos_spacing(input_data, scale=1):
    print(input_data)
    shifted_input_data = input_data - (input_data[-1] + input_data[0])/2
    nondim_input_data = (shifted_input_data - shifted_input_data[0]) / (shifted_input_data[-1] - shifted_input_data[0])
    theta = np.pi * nondim_input_data
    nondim_output = scale*np.cos(theta)
    cos_space_val = nondim_output * (input_data[0] - input_data[-1])/2/scale + (input_data[-1] + input_data[0])/2

    print(nondim_input_data)
    print(theta)
    print(nondim_output)

    return cos_space_val


if __name__ == '__main__':
    n = 9
    input_data = np.linspace(-4, 0, n)
    output_data = cos_spacing(input_data)

    import matplotlib.pyplot as plt
    for i in range(n):
        plt.plot([1., 1.1], [input_data[i], output_data[i]], 'k--')
    plt.plot([1] * n, input_data, '-o', label='linspace')
    plt.plot([1.1] * n, output_data, '-o', label='cos space')
    plt.xlim((.9, 1.2))
    plt.show()