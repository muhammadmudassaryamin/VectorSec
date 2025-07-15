import requests
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to get embeddings from Ollama
def get_ollama_embeddings(texts, model="llama3.2:latest", endpoint="http://localhost:11434/api/embeddings"):
    embeddings = []
    for text in texts:
        response = requests.post(endpoint, json={"model": model, "prompt": text})
        if response.status_code == 200:
            embedding = response.json().get("embedding")
            embeddings.append(embedding)
        else:
            print(f"Error fetching embedding for '{text}': {response.status_code}")
            embeddings.append(None)
    return embeddings

# Initialize lists to store texts and embeddings
texts = []
embeddings = []

# Set up interactive plotting
plt.ion()  # Turn on interactive mode
fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Initialize PCA
pca_2d = PCA()
pca_3d = PCA()

print("Enter words or sentences to visualize their embeddings (type 'exit' to finish):")

while True:
    # Get user input
    user_input = input("Enter text: ").strip()
    if user_input.lower() == 'exit':
        break
    
    # Add text to list
    texts.append(user_input)
    
    # Fetch embedding for the new text
    print(f"Fetching embedding for '{user_input}'...")
    new_embedding = get_ollama_embeddings([user_input])
    
    if new_embedding[0] is not None:
        embeddings.append(new_embedding[0])
        
        # Only proceed if we have valid embeddings
        if len(embeddings) >= 2:  # PCA needs at least 2 points
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings)
            n_samples = embeddings_array.shape[0]
            
            # Apply PCA for 2D (use min of 2 and n_samples)
            n_components_2d = min(2, n_samples)
            pca_2d.n_components = n_components_2d
            embeddings_2d = pca_2d.fit_transform(embeddings_array)
            
            # Clear previous 2D plot
            ax_2d.clear()
            
            # Plot 2D
            for i, text in enumerate(texts):
                if n_components_2d == 1:
                    # For 1 component, plot on a single axis
                    ax_2d.scatter(embeddings_2d[i, 0], 0, marker='o')
                    ax_2d.text(embeddings_2d[i, 0] + 0.02, 0 + 0.02, text, fontsize=9)
                else:
                    ax_2d.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], marker='o')
                    ax_2d.text(embeddings_2d[i, 0] + 0.02, embeddings_2d[i, 1] + 0.02, text, fontsize=9)
            ax_2d.set_title(f"{n_components_2d}D Visualization of LLM Vector Space (PCA)")
            ax_2d.set_xlabel("Component 1")
            ax_2d.set_ylabel("Component 2" if n_components_2d > 1 else "")
            ax_2d.grid(True)
            
            # Redraw 2D plot
            fig_2d.canvas.draw()
            fig_2d.canvas.flush_events()
            
            # Apply PCA for 3D (only if we have at least 3 samples)
            if n_samples >= 3:
                n_components_3d = min(3, n_samples)
                pca_3d.n_components = n_components_3d
                embeddings_3d = pca_3d.fit_transform(embeddings_array)
                
                # Clear previous 3D plot
                ax_3d.clear()
                
                # Plot 3D
                for i, text in enumerate(texts):
                    if n_components_3d == 1:
                        ax_3d.scatter(embeddings_3d[i, 0], 0, 0, marker='o')
                        ax_3d.text(embeddings_3d[i, 0] + 0.02, 0 + 0.02, 0, text, size=8)
                    elif n_components_3d == 2:
                        ax_3d.scatter(embeddings_3d[i, 0], embeddings_3d[i, 1], 0, marker='o')
                        ax_3d.text(embeddings_3d[i, 0] + 0.02, embeddings_3d[i, 1] + 0.02, 0, text, size=8)
                    else:
                        ax_3d.scatter(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], marker='o')
                        ax_3d.text(embeddings_3d[i, 0] + 0.02, embeddings_3d[i, 1] + 0.02, embeddings_3d[i, 2], text, size=8)
                ax_3d.set_title(f"{n_components_3d}D Visualization of LLM Vector Space (PCA)")
                ax_3d.set_xlabel("Component 1")
                ax_3d.set_ylabel("Component 2" if n_components_3d > 1 else "")
                ax_3d.set_zlabel("Component 3" if n_components_3d > 2 else "")
                
                # Redraw 3D plot
                fig_3d.canvas.draw()
                fig_3d.canvas.flush_events()
            else:
                print("Need at least 3 valid embeddings for 3D visualization.")
        else:
            print("Need at least 2 valid embeddings for visualization.")
    else:
        print(f"Skipping visualization for '{user_input}' due to embedding error.")

# Turn off interactive mode and save final plots
plt.ioff()
if len(embeddings) >= 2:
    fig_2d.savefig("vectorspace_2d.png")
    if len(embeddings) >= 3:
        fig_3d.savefig("vectorspace_3d.png")
        print("Final plots saved as 'vectorspace_2d.png' and 'vectorspace_3d.png'.")
    else:
        print("Final 2D plot saved as 'vectorspace_2d.png'. Not enough embeddings for 3D plot.")
else:
    print("Not enough valid embeddings to save plots.")

plt.show()