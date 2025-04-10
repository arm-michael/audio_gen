import pkg_resources
import torch
import torchaudio
import os
import gc
import time
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Force device execution
device = "mps"
print(f"Using device: {device}")

# Enable memory-efficient CPU execution if available
torch.set_num_threads(os.cpu_count())  # Use all available CPU cores

# **LOAD MODEL ONCE AT START**
print("Loading model...")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
model = model.to(device).to(torch.float32)  # Ensure correct device and dtype
print("Model loaded successfully.")

def generate_audio(model, prompt, tempo, duration, steps, cfg_scale, gen_count):
    start_time = time.time()

    try:
        # **Run garbage collection every 3rd generation to reduce overhead**
        if gen_count % 3 == 0:
            gc.collect()
            print(f"Memory cleared at generation {gen_count}")

        # Set up conditioning
        conditioning = [{
            "prompt": prompt,
            "tempo": tempo,
            "seconds_start": 0,
            "seconds_total": duration
        }]

        print(f"Generating audio for: {prompt}")

        # Generate audio with optimized parameters
        with torch.inference_mode():
            output = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",  # Consider changing for speed
                device=device
            )

        output = rearrange(output, "b d n -> d (b n)")

        # Ensure output tensor is on the correct device and dtype
        output = output.to(device).to(torch.float32)

        # Normalize and convert to int16
        output = (output.div(torch.max(torch.abs(output)))  
                 .clamp(-1, 1)
                 .mul(32767)
                 .to(torch.int16)
                 .cpu())

        # Save the generated audio as a .wav file
        outFile = f"{prompt}.wav"
        torchaudio.save(outFile, output, sample_rate)

        print(f"Generated audio saved to: {outFile}")
        return outFile

    except Exception as e:
        print(f"Error during audio generation: {str(e)}")
        raise

    finally:
        elapsed_time = time.time() - start_time
        if elapsed_time > 300:
            print(f"Warning: Audio generation took too long ({elapsed_time:.2f} seconds).")
            raise TimeoutError("Audio generation exceeded time limit.")

# **Loop for User Input**
if __name__ == "__main__":
    continue_generating = True
    gen_count = 0  # Track number of generations

    while continue_generating:
        try:
            print("Enter a prompt for generating audio:")
            prompt = str(input())

            print("Enter a tempo for the audio:")
            tempo = int(input())

            gen_count += 1  # Increment generation count
            output_file = generate_audio(
                model,                  # model
                prompt,                 # prompt
                tempo,                  # tempo (bpm)
                duration=10,            # duration (40 default)
                steps=7,                # consider changing for speed (60 default)
                cfg_scale=1,            # consider changing for speed (20 default)
                gen_count=gen_count     # pass the generation count
            )

            print("Would you like to generate more audio? (yes/no)")
            user_response = str(input().strip().lower()) 
            continue_generating = user_response == 'yes'

            if not continue_generating:
                print("Exiting audio generation.")

        except ValueError:
            print("Invalid input. Please enter a valid tempo (integer).")
        except TimeoutError:
            print("Audio generation timed out.")
        except Exception as e:
            print(f"An error occurred: {e}")
            continue_generating = user_response == 'yes'