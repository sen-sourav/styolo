#import streamlit as st
#from audiorecorder import audiorecorder
#from app.model_loader import load_model
#from app.model_inference import model_inference
#import os
#import pathlib
#
## Load model when Streamlit server starts
#model = load_model()
#
#def count_wav_files(directory):
#    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name)) and name.endswith('.wav')])
#
#st.title("Add piano accompaniment")
#
#input_file_dir = str(pathlib.Path("./uploads").resolve())
#output_dir = str(pathlib.Path("./output").resolve())
#
##audio = audiorecorder("Click to record", "Click to stop recording")
#audio = audiorecorder()
## Clear previous audio
#st.empty()
#
#if len(audio) > 0:
#
#    # Generate new filename
#    filename = f"audio_{(count_wav_files(input_file_dir))+1}.wav"
#    input_file_path = f"{input_file_dir}/{filename}"
#    out_audio_path = f'{output_dir}/{filename.split(".")[0]}_final_mix.wav'
#
#    st.write(f"Input:")
#    # To play audio in frontend:
#    st.audio(audio.export().read())
#
#    # To save audio to a file, use pydub export method:
#    audio.export(input_file_path, format="wav")
#
#    # To get audio properties, use pydub AudioSegment properties:
#    #st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
#
#    # Perform model inference when user records sound
#    model_inference(model, input_file_path, output_dir)
#    
#    st.write(f"Ouput:")
#    st.audio(out_audio_path, format="wav", loop=False)
#
#
#    # Display download buttons
#    with open(input_file_path, "rb") as file:
#        st.download_button("Download Original Audio", file, filename)
#
#    with open(out_audio_path, "rb") as file:
#        st.download_button("Download Processed Audio", file, out_audio_path.split("/")[-1])





import streamlit as st
from audiorecorder import audiorecorder
from app.model_loader import load_model
from app.model_inference import model_inference
import os
import pathlib
from streamlit_js_eval import streamlit_js_eval

# Load model when Streamlit server starts
model = load_model()

def count_wav_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name)) and name.endswith('.wav')])

def clear_session_state():
    """Clear all session state variables."""
    for key in st.session_state.keys():
        del st.session_state[key]


    # Simulate a rerun by resetting the query parameters
    st.query_params.clear()

# Title
st.title("add piano")

# Directories
input_file_dir = str(pathlib.Path("./uploads").resolve())
output_dir = str(pathlib.Path("./output").resolve())


# Terms of Service agreement
if "tos_agreed" not in st.session_state:
    st.session_state["tos_agreed"] = False

if st.checkbox('I agree to the Terms of Service', value=st.session_state["tos_agreed"]):
    st.session_state["tos_agreed"] = True
else:
    with st.expander("Read Terms of Service"):
        st.markdown("""
            ### Terms of Service

                - Privacy Policy:
                    * We respect your privacy and are committed to protecting your personal
                information. Our privacy policy explains how we collect, use, and safeguard 
                your data.
                    * Information Collection: We collect data necessary for providing our 
                services such as user_names, vocal recordings, and user-generated content.
                    * Data Usage: We use your data to enhance our services, improve user 
                experience, and communicate with you.
                    * Data Protection: We implement robust security measures to protect your 
                data from unauthorized access, disclosure, or use.
                    * Data Sharing: We DO NOT share your data with third parties, except as 
                required by law.

                - Disclaimer:

                    * Content Ownership:  You retain ownership of your vocal recordings and 
                user-generated content. By using YOLO MUSIC, you grant us a non-exclusive 
                license to process and use your content for providing our services.
                    * Intellectual Property:  YOLO MUSIC owns all rights to its software, 
                algorithms, and generated background music.
                    * Limitation of Liability: In no event shall YOLO MUSIC or its 
                OWNERS and DEVELOPERS be liable for any damages, losses, or expenses arising
                from our services on discord.
                    * Warranty Disclaimer:  Our services are provided "as is" and 
                "as available," without warranties of any kind, express or implied.
                    * Copyright Compliance: You acknowledge that you are solely responsible
                for ensuring that your use of YOLO MUSIC does not violate any copyright laws
                and regulations. You must not upload any copyrighted materials, such as music
                or vocals from artists or others members, without proper authorization or licenses.
                YOLO MUSIC or its OWNERS and DEVELOPERS are not responsible for any copyright
                infringement resulting from your actions.


                By using YOLO MUSIC, you acknowledge that you have read, understood, and agreed
                to our Privacy Policy and Disclaimer.
                If you have any questions or concerns, please contact us at 
                https://discord.com/channels/1269795281914957917/1269852340488372254
            """)



if st.button("refresh"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


# Add a "Clear" button at the end
#if st.button("Clear"):
#    clear_session_state()


# Record audio
#audio = audiorecorder("Click to record", "Click to stop recording")

st.write("Click the button below to start recording vocals to the song")
audio = audiorecorder()
duration = 30 #in seconds
# Check if audio is recorded
if len(audio) > 0:
    audio = audio[:duration * 1000]
    # Generate new filename
    filename = f"audio_{(count_wav_files(input_file_dir))+1}.wav"
    input_file_path = f"{input_file_dir}/{filename}"
    out_audio_path = f'{output_dir}/{filename.split(".")[0]}_final_mix.wav'

    # Display input audio
    st.write(f"Input:")
    st.audio(audio.export().read())

    # Save audio to file
    audio.export(input_file_path, format="wav")

    # Perform model inference when user records sound
    model_inference(model, input_file_path, output_dir)
    
    # Display output audio
    st.write(f"Output:")
    st.audio(out_audio_path, format="wav", loop=False)

    # Display download buttons
    with open(input_file_path, "rb") as file:
        st.download_button("Download Input Audio", file, filename)

    with open(out_audio_path, "rb") as file:
        st.download_button("Download Output Audio", file, out_audio_path.split("/")[-1])

