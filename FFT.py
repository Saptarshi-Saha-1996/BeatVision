import numpy as np
import matplotlib.pyplot as plt

def signal_to_matrix(signal):
    """
    Converts a 1D signal into a matrix representation using Fourier Transform.
    
    Parameters:
    signal (np.ndarray): The input 1D signal.
    
    Returns:
    np.ndarray: A matrix where each row contains the real and imaginary parts of a Fourier coefficient.
    """
    # Compute the Fourier Transform
    fourier_transform = np.fft.fft(signal)
    
    # Create a matrix with real and imaginary parts
    matrix = np.vstack((fourier_transform.real, fourier_transform.imag)).T
    return matrix

def matrix_to_signal(matrix):
    """
    Converts a matrix representation back into the original 1D signal using inverse Fourier Transform.
    
    Parameters:
    matrix (np.ndarray): A matrix where each row contains the real and imaginary parts of a Fourier coefficient.
    
    Returns:
    np.ndarray: The reconstructed 1D signal.
    """
    # Reconstruct the Fourier coefficients from the matrix
    fourier_transform = matrix[:, 0] + 1j * matrix[:, 1]
    
    # Perform the inverse Fourier Transform to get the original signal back
    reconstructed_signal = np.fft.ifft(fourier_transform).real
    return reconstructed_signal

def plot_matrix(matrix):
    """
    Plots the matrix as a heatmap.
    
    Parameters:
    matrix (np.ndarray): The matrix to be plotted.
    """
    plt.figure(figsize=(12, 6))

    # Calculate mean and standard deviation for better visualization
    mean_real = np.mean(matrix[:, 0])
    std_real = np.std(matrix[:, 0])
    mean_imag = np.mean(matrix[:, 1])
    std_imag = np.std(matrix[:, 1])

    vmin_real = mean_real - 2 * std_real
    vmax_real = mean_real + 2 * std_real
    vmin_imag = mean_imag - 2 * std_imag
    vmax_imag = mean_imag + 2 * std_imag

    # Plot real parts
    plt.subplot(2, 1, 1)
    plt.imshow(matrix[:, 0:1].T, aspect='auto', cmap='viridis', vmin=vmin_real, vmax=vmax_real)
    plt.colorbar(label='Real Part')
    plt.title('Real Parts of Fourier Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Real Part')

    # Plot imaginary parts
    plt.subplot(2, 1, 2)
    plt.imshow(matrix[:, 1:2].T, aspect='auto', cmap='viridis', vmin=vmin_imag, vmax=vmax_imag)
    plt.colorbar(label='Imaginary Part')
    plt.title('Imaginary Parts of Fourier Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Imaginary Part')

    plt.tight_layout()
    plt.savefig('matrix_heatmap.png')
    plt.show()

def find_min_max(matrix):
    """
    Finds the minimum and maximum values for both the real and imaginary parts of the matrix.
    
    Parameters:
    matrix (np.ndarray): The matrix to analyze.
    
    Returns:
    tuple: Min and max values for real and imaginary parts (min_real, max_real, min_imag, max_imag).
    """
    min_real = np.min(matrix[:, 0])
    max_real = np.max(matrix[:, 0])
    min_imag = np.min(matrix[:, 1])
    max_imag = np.max(matrix[:, 1])
    return min_real, max_real, min_imag, max_imag

# Example usage
if __name__ == "__main__":
    # Given example signal and sampling rate
    #signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    sampling_rate = 300 # Example sampling rate (samples per second)
    

    filename = "A00019"
    filepath = "/home/saptarshi/Research/CF_Explanation/ISI_2D_to_1D_data/" + str(filename) + ".npy"
    signal = np.load(filepath)
    # Convert signal to matrix
    matrix = signal_to_matrix(signal)
    print("Matrix Representation:\n", matrix)
    
    # Convert matrix back to signal
    reconstructed_signal = matrix_to_signal(matrix)
    print("Reconstructed Signal:", reconstructed_signal)
    
    # Plot the original and reconstructed signals for comparison
    plt.figure(figsize=(12, 6))
    
    # Original Signal
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Reconstructed Signal
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_signal, label='Reconstructed Signal', linestyle='--')
    plt.title('Reconstructed Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('original_vs_reconstructed_signal.png')
    plt.show()
    
    # Find min and max values for real and imaginary parts
    min_real, max_real, min_imag, max_imag = find_min_max(matrix)
    print(f"Min Real: {min_real}, Max Real: {max_real}")
    print(f"Min Imaginary: {min_imag}, Max Imaginary: {max_imag}")
    
    # Plot the matrix with value limits for better visualization
    plot_matrix(matrix)
