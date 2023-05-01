import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from models import *
from torch.autograd import Variable

from differential_evolution import differential_evolution


def perturb_image(xs, img):
	if xs.ndim < 2:
		xs = np.array([xs])
	batch = len(xs)
	imgs = img.repeat(batch, 1, 1, 1)
	xs = xs.astype(int)

	count = 0
	for x in xs:
		pixels = np.split(x, len(x)/5)
		
		for pixel in pixels:
			x_pos, y_pos, r, g, b = pixel
			imgs[count, 0, x_pos, y_pos] = (r/255.0-0.4914)/0.2023
			imgs[count, 1, x_pos, y_pos] = (g/255.0-0.4822)/0.1994
			imgs[count, 2, x_pos, y_pos] = (b/255.0-0.4465)/0.2010
		count += 1

	return imgs

def predict_classes(xs, img, target_calss, net, minimize=True):
	imgs_perturbed = perturb_image(xs, img.clone())
	with torch.no_grad():
		input = Variable(imgs_perturbed).cuda()
	predictions = F.softmax(net(input),dim=1).data.cpu().numpy()[:, target_calss]

	return predictions if minimize else 1 - predictions

def attack_success(x, img, target_calss, net, targeted_attack=False, verbose=False):

	attack_image = perturb_image(x, img.clone())
	with torch.no_grad():
		input = Variable(attack_image).cuda()
	confidence = F.softmax(net(input),dim=1).data.cpu().numpy()[0]
	predicted_class = np.argmax(confidence)

	if (verbose):
		print("Confidence: %.4f"%confidence[target_calss])
	if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
		return True

def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
	# img: 1*3*W*H tensor
	# label: a number

	targeted_attack = target is not None
	target_calss = target if targeted_attack else label

	bounds = [(0,32), (0,32), (0,255), (0,255), (0,255)] * pixels

	popmul = int(max(1, popsize/len(bounds)))

	predict_fn = lambda xs: predict_classes(
		xs, img, target_calss, net, target is None)
	callback_fn = lambda x, convergence: attack_success(
		x, img, target_calss, net, targeted_attack, verbose)

	inits = np.zeros([popmul*len(bounds), len(bounds)])
	for init in inits:
		for i in range(pixels):
			init[i*5+0] = np.random.random()*32
			init[i*5+1] = np.random.random()*32
			init[i*5+2] = np.random.normal(128,127)
			init[i*5+3] = np.random.normal(128,127)
			init[i*5+4] = np.random.normal(128,127)

	attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
		recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

	attack_image = perturb_image(attack_result.x, img)
	with torch.no_grad():
		attack_var = Variable(attack_image).cuda()
	predicted_probs = F.softmax(net(attack_var),dim=1).data.cpu().numpy()[0]

	predicted_class = np.argmax(predicted_probs)

	if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
		return 1, attack_result.x.astype(int)
	return 0, [None]

def attack_all(net, loader, samples=100, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):

	num_sample = 0
	success = 0

	samples = len(loader) if samples == -1 else samples

	for batch_idx, (img_var, target) in enumerate(loader):

		pred = net(img_var)
		prior_probs = F.softmax(pred,dim=1)
		_, indices = torch.max(prior_probs, 1)
		
		num_sample += 1
		if target[0] != indices.data.cpu()[0]:
			continue
		
		target = target.numpy()

		flag, x = attack(input, target[0], net, None, pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)
		success += flag

	
		if num_sample == samples:
			break


	success_rate = float(success)/num_sample
	print("success rate: %.4f (%d/%d)" % (success_rate, success, num_sample))
        # [(x,y) = (%d,%d) and (R,G,B)=(%d,%d,%d)]"%(
		# success_rate, success, num_sample += 1, x[0],x[1],x[2],x[3],x[4]))


	return success_rate

def main():

	net = <your net> & load weights
	testloader = <your dtlder>, bs=1

	# samples = -1: attack for all test images
	results = attack_all(net, testloader, pixels=1, samples=-1)
	print("Final success rate: %.4f"%results)


if __name__ == '__main__':
	main()