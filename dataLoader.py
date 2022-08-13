'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal

# from aukit.audio_noise_remover import remove_noise
# from aukit.audio_normalizer import tune_volume, remove_silence

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-3] not in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			self.noiselist[file.split('/')[-3]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index])
		# audio = remove_noise(wav=audio, sr=16000)
		# audio = remove_silence(audio, max_silence_ms=1000)
		# audio = tune_volume(audio, target_dBFS=-10)
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)

		# Data Augmentation
#		augtype = random.randint(1,5)
#
#		if augtype == 1: # Reverberation
#			audioaug = self.add_rev(audio)
#		elif augtype == 2: # Babble
#			audioaug = self.add_noise(audio,'speech')
#		elif augtype == 3: # Music
#			audioaug = self.add_noise(audio,'music')
#		elif augtype == 4: # Noise
#			audioaug = self.add_noise(audio,'noise')
#		elif augtype == 5: # Television noise
#			audioaug = self.add_noise(audio,'speech')
#			audioaug = self.add_noise(audioaug,'music')
		audioaug1 = self.add_rev(audio)
		audioaug2 = self.add_noise(audio, 'speech')
		audioaug3 = self.add_noise(audio, 'music')
		audioaug4 = self.add_noise(audio, 'noise')

		return torch.FloatTensor(audioaug1[0]),torch.FloatTensor(audioaug2[0]),torch.FloatTensor(audioaug3[0]),torch.FloatTensor(audioaug4[0]),torch.FloatTensor(audio[0]), self.data_label[index]


#		return torch.FloatTensor(audioaug[0]),torch.FloatTensor(audio[0]), self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file = random.choice(self.rir_files)
		rir, sr = soundfile.read(rir_file)
		rir = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise = self.numnoise[noisecat]
		num = random.randint(numnoise[0],numnoise[1])
		if(len(self.noiselist[noisecat])<num):
			num = len(self.noiselist[noisecat]) -1
		noiselist = random.sample(self.noiselist[noisecat],num)
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio