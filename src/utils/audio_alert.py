"""
Audio Alert utility using Pygame for Driver Drowsiness System.
Handles initializing the audio mixer, playing alarm tracks, and generating fallback synthesized beeps.
"""

import os
import math
import array
import threading
import pygame

class AudioAlertSystem:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.mixer_initialized = False
        self.alarm_sound = None
        self.fallback_beep = None
        self.sound_channel = None
        self.is_playing = False
        
        try:
            pygame.mixer.init()
            self.mixer_initialized = True
            self._create_fallback_beep()
        except Exception as e:
            try:
                os.environ["SDL_AUDIODRIVER"] = "dummy"
                pygame.mixer.init()
                self.mixer_initialized = True
                self._create_fallback_beep()
            except Exception as e2:
                print("[AudioAlertSystem] Warning: Failed to initialize pygame mixer: {}".format(e2))
            
    def _create_fallback_beep(self):
        """Generates an 880Hz alert tone in-memory if no custom alarm file is provided or found."""
        if not self.mixer_initialized:
            return
            
        try:
            duration = 0.5 # half-second beep
            frequency = 880.0
            num_samples = int(self.sample_rate * duration)
            
            buf = array.array('h', [0] * num_samples)
            for i in range(num_samples):
                t = i / self.sample_rate
                # Generate sine wave with peak amplitude
                val = int(30000 * math.sin(2 * math.pi * frequency * t))
                buf[i] = val
                
            self.fallback_beep = pygame.mixer.Sound(buffer=bytes(buf))
        except Exception as e:
            print("[AudioAlertSystem] Failed to generate fallback beep: {}".format(e))

    def load_alarm_file(self, file_path):
        """Loads a WAV or MP3 alert file."""
        if not self.mixer_initialized:
            return False
            
        try:
            self.alarm_sound = pygame.mixer.Sound(file_path)
            return True
        except Exception as e:
            print("[AudioAlertSystem] Failed to load alarm file '{}': {}".format(file_path, e))
            return False

    def play_alert(self, loop=True):
        """Plays the loaded alarm or the fallback beep."""
        if not self.mixer_initialized:
            return
            
        if self.is_playing:
            return
            
        try:
            if os.environ.get("SDL_AUDIODRIVER") == "dummy":
                self.is_playing = True
                return

            sound = self.alarm_sound if self.alarm_sound else self.fallback_beep
            if sound:
                loops = -1 if loop else 0
                self.sound_channel = sound.play(loops=loops)
                self.is_playing = True
        except Exception as e:
            print("[AudioAlertSystem] Error playing alert sound: {}".format(e))

    def stop_alert(self):
        """Stops the active alert play."""
        if not self.mixer_initialized or not self.is_playing:
            return
            
        try:
            if self.sound_channel:
                self.sound_channel.stop()
            elif self.alarm_sound:
                self.alarm_sound.stop()
            elif self.fallback_beep:
                self.fallback_beep.stop()
        except Exception as e:
            print("[AudioAlertSystem] Error stopping alert sound: {}".format(e))
        finally:
            self.is_playing = False
            
    def shutdown(self):
        """Stops any sound and uninitializes the pygame mixer."""
        self.stop_alert()
        if self.mixer_initialized:
            try:
                pygame.mixer.quit()
            except:
                pass
