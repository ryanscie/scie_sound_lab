# scie_sound_lab
Interactive demos for our school Sound Technology Club

## Active Noise Cancelling Lab

From the repository root, run:

```sh
python3 -m http.server 8765 --bind localhost
```

Open http://localhost:8765/lessons/03-noisecancelling/ in a current browser.
The lab uses JavaScript modules and AudioWorklet, so use localhost or HTTPS
instead of opening the HTML as a local file.

Choose **Start listening** to hear synthesized tones and noise. The main
simulator follows the ANC switch and the noise, wanted sound, strength and delay
controls. The separate 120 Hz timing test shares the strength control, but uses
its own phase setting and ignores the main ANC switch. Select its listening
source or choose **Listen to this timing test**. Playback volume and mute affect
only listening; **Pause waves** pauses only the animation. Switching tabs stops
playback; it resumes only when you start listening again.

Run the signal and audio-rendering checks with Node.js:

```sh
node --test lessons/03-noisecancelling/tests/audio.test.mjs
```
