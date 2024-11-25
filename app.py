import json
import os
import random
from openai import OpenAI
from gtts import gTTS
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy.editor import AudioFileClip, concatenate_videoclips, VideoClip, concatenate_audioclips, CompositeAudioClip
from moviepy.video.io.bindings import mplfig_to_npimage

client = OpenAI(
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    api_key='API KEY OF GEMNI'
)

prompt = '''
give the below details in a json format based on data given by user
{
    "video_name": "", # the name can be if we convert chart to video, ex: abc-analysis.mp4
    "type": "", # can be one of these: pie, bar, line choose one of this and create an great chart
    "labels": [] # extract label names from data
    "sizes": [] # extract values of perticular label from data no need in line chart
    "x_values": [] # extract values of perticular label from data only in line chart
    "y_values": [] # extract values of perticular label from data only in line chart
    "colors": [] # color for the slice of that label in a chart try to make this colorfull bg will be light white, as bg will similar to white so dont use very light colors
    "background": "", # bg color of a chart make good combanition with the bg and colors
    "explaination": [], # list of text to explain the slice of a perticular label, explain like a story telling
    "explaination_before": "", # an text to speak before the chart explain
    "explaination_after": "", # an text to speak after the chart explaination
    "background_music": "", # an background music can be one of: '''+', '.join(os.listdir('songs'))+'''
}
'''
response = client.chat.completions.create(
    model="gemini-1.5-flash",
    n=1,
    messages=[
        {"role": "system", "content": prompt},
        {
            "role": "user",
            # "content": "20% of users own an iPhone, 50% own a Samsung, and the rest own a variety of brands"
            "content": '''Supermarket Sales Data							
							
						Tax	10%
							
Order No	Order Date	Customer Name	Ship Date	Retail Price (USD)	Order Quantity	Tax (USD)	Total (USD)
1001	01/01/2024	John Smith	03/01/2024	49.99	2	9.998	109.978
1002	01/01/2024	Jane Doe	04/01/2024	29.99	1	2.999	32.989
1003	02/01/2024	Michael Johnson	07/01/2024	99.99	3	29.997	329.967
1004	02/01/2024	Emily Brown	03/01/2024	19.99	4	7.996	87.956
1005	03/01/2024	David Wilson	08/01/2024	149.99	1	14.999	164.989
1006	03/01/2024	Lisa Taylor	06/01/2024	79.99	2	15.998	175.978
1007	04/01/2024	Daniel Martinez	06/01/2024	39.99	3	11.997	131.967
1008	04/01/2024	Sarah Anderson	09/01/2024	69.99	2	13.998	153.978
1009	05/01/2024	Christopher Thomas	06/01/2024	89.99	1	8.999	98.989
1010	05/01/2024	Kimberly Garcia	08/01/2024	199.99	1	19.999	219.989
1011	06/01/2024	William Hernandez	07/01/2024	29.99	5	14.995	164.945
1012	06/01/2024	Melissa Lopez	08/01/2024	79.99	2	15.998	175.978
1013	07/01/2024	Richard Perez	09/01/2024	49.99	3	14.997	164.967
1014	07/01/2024	Jessica Gonzalez	12/01/2024	129.99	1	12.999	142.989
1015	08/01/2024	Matthew Wilson	13/01/2024	19.99	4	7.996	87.956
1016	08/01/2024	Amanda Martinez	12/01/2024	149.99	1	14.999	164.989
1017	09/01/2024	James Johnson	14/01/2024	69.99	2	13.998	153.978
1018	09/01/2024	Laura Brown	12/01/2024	39.99	3	11.997	131.967
1019	10/01/2024	Daniel Smith	11/01/2024	199.99	1	19.999	219.989
1020	10/01/2024	Jennifer Davis	14/01/2024	29.99	5	14.995	164.945
1021	11/01/2024	Michael Garcia	14/01/2024	79.99	2	15.998	175.978
1022	11/01/2024	Amy Hernandez	15/01/2024	49.99	3	14.997	164.967
1023	12/01/2024	Christopher Rodriguez	17/01/2024	129.99	1	12.999	142.989
1024	12/01/2024	Jessica Martinez	17/01/2024	19.99	4	7.996	87.956
1025	13/01/2024	David Wilson	17/01/2024	149.99	1	14.999	164.989
1026	13/01/2024	Sarah Smith	14/01/2024	69.99	2	13.998	153.978
1027	14/01/2024	Matthew Johnson	18/01/2024	39.99	3	11.997	131.967
1028	14/01/2024	Emily Davis	19/01/2024	199.99	1	19.999	219.989
1029	15/01/2024	Daniel Wilson	19/01/2024	29.99	5	14.995	164.945
1030	15/01/2024	Jennifer Martinez	19/01/2024	79.99	2	15.998	175.978
1031	16/01/2024	Michael Smith	19/01/2024	49.99	3	14.997	164.967
1032	16/01/2024	Jessica Johnson	17/01/2024	129.99	1	12.999	142.989
1033	17/01/2024	David Brown	19/01/2024	19.99	4	7.996	87.956
1034	17/01/2024	Sarah Garcia	19/01/2024	149.99	1	14.999	164.989
1035	18/01/2024	Matthew Hernandez	23/01/2024	69.99	2	13.998	153.978
1036	18/01/2024	Emily Rodriguez	19/01/2024	39.99	3	11.997	131.967
1037	19/01/2024	Daniel Davis	20/01/2024	199.99	1	19.999	219.989
1038	19/01/2024	Jennifer Smith	22/01/2024	29.99	5	14.995	164.945
1039	20/01/2024	Michael Johnson	24/01/2024	79.99	2	15.998	175.978
1040	20/01/2024	Jessica Martinez	21/01/2024	49.99	3	14.997	164.967
1041	21/01/2024	David Wilson	26/01/2024	129.99	1	12.999	142.989
1042	21/01/2024	Sarah Johnson	23/01/2024	19.99	4	7.996	87.956
1043	22/01/2024	Matthew Garcia	25/01/2024	149.99	1	14.999	164.989
1044	22/01/2024	Emily Brown	25/01/2024	69.99	2	13.998	153.978
1045	23/01/2024	Daniel Hernandez	27/01/2024	39.99	3	11.997	131.967
1046	23/01/2024	Jennifer Davis	26/01/2024	199.99	1	19.999	219.989
1047	24/01/2024	Michael Martinez	25/01/2024	29.99	5	14.995	164.945
1048	24/01/2024	Jessica Wilson	26/01/2024	79.99	2	15.998	175.978
1049	25/01/2024	David Rodriguez	30/01/2024	49.99	3	14.997	164.967
1050	25/01/2024	Sarah Gonzalez	27/01/2024	129.99	1	12.999	142.989
1051	26/01/2024	Matthew Smith	29/01/2024	19.99	4	7.996	87.956
1052	26/01/2024	Emily Johnson	28/01/2024	149.99	1	14.999	164.989
1053	27/01/2024	Daniel Brown	31/01/2024	69.99	2	13.998	153.978
1054	27/01/2024	Jennifer Hernandez	31/01/2024	39.99	3	11.997	131.967
1055	28/01/2024	Michael Davis	29/01/2024	199.99	1	19.999	219.989
1056	28/01/2024	Jessica Smith	29/01/2024	29.99	5	14.995	164.945
1057	29/01/2024	David Martinez	30/01/2024	79.99	2	15.998	175.978
1058	29/01/2024	Sarah Johnson	31/01/2024	49.99	3	14.997	164.967
1059	30/01/2024	Matthew Garcia	31/01/2024	129.99	1	12.999	142.989
1060	30/01/2024	Emily Brown	01/02/2024	19.99	4	7.996	87.956
1061	31/01/2024	Daniel Hernandez	02/02/2024	149.99	1	14.999	164.989
1062	31/01/2024	Jennifer Davis	04/02/2024	69.99	2	13.998	153.978
1063	01/02/2024	Michael Martinez	02/02/2024	39.99	3	11.997	131.967
1064	01/02/2024	Jessica Wilson	05/02/2024	199.99	1	19.999	219.989
1065	01/02/2024	David Rodriguez	03/02/2024	29.99	5	14.995	164.945
1066	02/02/2024	Sarah Gonzalez	06/02/2024	79.99	2	15.998	175.978
1067	03/02/2024	Matthew Smith	06/02/2024	49.99	3	14.997	164.967
1068	04/02/2024	Emily Johnson	05/02/2024	129.99	1	12.999	142.989
1069	04/02/2024	Daniel Brown	08/02/2024	19.99	4	7.996	87.956
1070	04/02/2024	Jennifer Hernandez	07/02/2024	149.99	1	14.999	164.989

'''
        }
    ],
    response_format={"type": "json_object"}
)

response = json.loads(response.choices[0].message.content)
print(response)

audio_files = []

os.makedirs('audio', exist_ok=True)
for i, txt in zip(range(len(response.get('explaination', []))), response.get('explaination', [])):
    tts = gTTS(text=txt, lang='en', slow=False)
    file_name = f'audio/{i}.mp3'
    tts.save(file_name)
    audio_files.append(file_name)

tts = gTTS(text=response.get('explaination_before'), lang='en', slow=False)
file_name = f'audio/before.mp3'
tts.save(file_name)
explaination_before = file_name

tts = gTTS(text=response.get('explaination_after'), lang='en', slow=False)
file_name = f'audio/after.mp3'
tts.save(file_name)
explaination_after = file_name

labels = response.get('labels') or []
sizes = response.get('sizes') or []
colors = response.get('colors') or []
x_values = response.get('x_values') or []
y_values = response.get('y_values') or []
background = response.get('background') or '#f4f4f9'

fig, ax = plt.subplots()


def pie_make_frame(current_segment, is_explaining=True):
    ax.clear()
    ax.axis('equal')

    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    explode = [0.1 if is_explaining and i ==
               current_segment else 0 for i in range(len(sizes))]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f"{int(pct)}%",
        startangle=90,
    )
    ax.set_title(
        f"Segment: {labels[current_segment]} - {sizes[current_segment]}%", fontsize=14, weight="bold")

    return mplfig_to_npimage(fig)


def pie_create_segment_clip(segment_index, audio_file, is_explaining=True):
    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration

    # Create a video clip
    video_clip = VideoClip(
        lambda t: pie_make_frame(segment_index, is_explaining),
        duration=duration
    ).set_audio(audio_clip)

    return video_clip


def bar_make_frame(current_segment, t, duration, is_explaining=True):
    """
    Generate a frame for the bar chart animation with blinking effect.
    :param current_segment: Index of the current bar being animated.
    :param t: Current time (in seconds).
    :param duration: Duration of the audio for the current segment.
    """
    ax.clear()  # Clear the current plot
    ax.set_facecolor(background)  # Set background color

    # Determine blink state (1 if visible, 0.4 if dimmed) based on time
    blink = 1.0 if int(t * 2) % 2 == 0 else 0.4

    # Create the bar chart
    bars = ax.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1)

    # Set alpha for each bar
    for i, bar in enumerate(bars):
        if not is_explaining:
            continue
        if i == current_segment:
            bar.set_alpha(blink)  # Blink the current bar
        else:
            bar.set_alpha(0.4)  # Dim other bars

    # Add value annotations above the bars
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{int(bar.get_height())}" if bar.get_height() > 0 else "",
            ha='center',
            fontsize=12,
            weight='bold'
        )

    # Add title showing the segment's label and value
    ax.set_title(
        f"Segment: {labels[current_segment]}",
        fontsize=14,
        weight="bold"
    )
    ax.set_ylim(0, max(sizes) + 10)  # Set consistent Y-axis limit
    ax.set_ylabel("Percentage")  # Add Y-axis label

    # Return the frame as an image
    return mplfig_to_npimage(fig)

# Function to create a video clip for each segment


def bar_create_segment_clip(segment_index, audio_file, is_explaining=True):
    """
    Create a video clip for a specific bar chart segment.
    :param segment_index: Index of the segment to highlight.
    :param audio_file: Path to the audio file for the segment.
    """
    # Load audio to determine duration
    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration

    # Create a video clip for the segment
    video_clip = VideoClip(
        # Pass current time and duration
        lambda t: bar_make_frame(segment_index, t, duration, is_explaining),
        duration=duration
    ).set_audio(audio_clip)

    return video_clip


def line_make_frame(current_segment, t, duration, is_explaining=True):
    """
    Generate a frame for the line chart animation with optional blinking effect.
    :param current_segment: Index of the current point being animated.
    :param t: Current time (in seconds).
    :param duration: Duration of the audio for the current segment.
    :param is_explaining: Whether the current point should blink. Default is True.
    """
    ax.clear()  # Clear the current plot
    ax.set_facecolor(background)  # Set background color

    # Determine blink state (1 if visible, 0.4 if dimmed) based on time and is_explaining
    blink = 1.0 if not is_explaining or int(t * 2) % 2 == 0 else 0.4

    # Plot the line connecting all points
    ax.plot(x_values, y_values, color='black', linewidth=2, zorder=1)

    # Plot all points using their respective colors
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.scatter(
            x, y,
            color=colors[i],  # Use the color from the colors array
            s=100,
            zorder=2,
            alpha=1.0 if i != current_segment else blink  # Highlight the current point
        )

    # Add value annotations slightly higher above each point
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        ax.text(
            x, y + 3,  # Adjusted to position labels slightly higher
            f"{y}",
            ha='center',
            fontsize=12,
            weight='bold',
            alpha=0.9
        )

    # Add title showing the segment's label and value
    ax.set_title(
        f"Segment: {labels[current_segment]} ({y_values[current_segment]})",
        fontsize=14,
        weight="bold"
    )
    ax.set_xlim(0.5, len(x_values) + 0.5)  # Set consistent X-axis range
    ax.set_ylim(0, max(y_values) + 10)  # Set consistent Y-axis range
    ax.set_xlabel("Segments")
    ax.set_ylabel("Values")

    # Return the frame as an image
    return mplfig_to_npimage(fig)


def line_create_segment_clip(segment_index, audio_file, is_explaining=True):
    """
    Create a video clip for a specific line chart segment.
    :param segment_index: Index of the segment to highlight.
    :param audio_file: Path to the audio file for the segment.
    :param is_explaining: Whether the current point should blink. Default is True.
    """
    # Load audio to determine duration
    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration

    # Create a video clip for the segment
    video_clip = VideoClip(
        # Pass current time and duration
        lambda t: line_make_frame(segment_index, t, duration, is_explaining),
        duration=duration
    ).set_audio(audio_clip)

    return video_clip


if response.get('type') == 'pie':
    ax.axis('equal')
    before_clip = pie_create_segment_clip(
        0, explaination_before, is_explaining=False)
    after_clip = pie_create_segment_clip(
        0, explaination_after, is_explaining=False)

    segment_clips = [before_clip]+[pie_create_segment_clip(i, audio_files[i])
                                   for i in range(len(labels))]+[after_clip]

if response.get('type') == 'bar':
    before_clip = bar_create_segment_clip(
        0, explaination_before, is_explaining=False)
    after_clip = bar_create_segment_clip(
        0, explaination_after, is_explaining=False)

    segment_clips = [before_clip]+[bar_create_segment_clip(i, audio_files[i])
                                   for i in range(len(labels))]+[after_clip]

if response.get('type') == 'line':
    before_clip = line_create_segment_clip(
        0, explaination_before, is_explaining=False)
    after_clip = line_create_segment_clip(
        0, explaination_after, is_explaining=False)

    segment_clips = [before_clip]+[line_create_segment_clip(i, audio_files[i])
                                   for i in range(len(labels))]+[after_clip]

final_clip = concatenate_videoclips(segment_clips, method="compose")

songs_dir = './songs'
song_files = [f for f in os.listdir(songs_dir) if f.endswith('.mp3')]
random_song = response.get('background_music') if os.path.exists(
    response.get('background_music')) else random.choice(song_files)

# 2. Load the video clip and song
original_audio = final_clip.audio  # The original audio from the video
audio_clip = AudioFileClip(os.path.join(songs_dir, random_song))

# 3. Check the length of the song and the video
video_duration = final_clip.duration
song_duration = audio_clip.duration

# 4. If the song is longer than the video, trim the song to the video length, else add extra 5 seconds of the song
if song_duration > video_duration:
    audio_clip = audio_clip.subclip(0, video_duration)
else:
    # Take first 5 seconds of the song for looping
    extra_audio = audio_clip.subclip(0, 5)
    audio_clip = concatenate_audioclips(
        [audio_clip, extra_audio])  # Add the 5 seconds at the end

# 5. Lower the volume of the song (e.g., 0.2 of the original volume)
audio_clip = audio_clip.volumex(0.05)

# 6. Combine the original audio with the background music
combined_audio = CompositeAudioClip([original_audio, audio_clip])

# 7. Set the combined audio to the final video
final_clip = final_clip.set_audio(combined_audio)
# 7. Export the final video
# final_clip.write_videofile('final_video_with_music.mp4', audio_codec='aac')


final_clip.write_videofile(response.get('video_name', 'video.mp4'), fps=24)
