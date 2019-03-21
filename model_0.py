import torch
import parser

# torch.zeros([2, 4], dtype = torch.int32)

def score(x, g, i, j):
	scores = torch.Tensor(
		[[[ 0,   1,   0,   0],
		 [  0,   0,   0,   1],
		 [  0,   1,   1,   1],
		 [  0,   80,   90,   0]],

		[[  0,   0,   0,   0],
		 [  0,   0,   0,   0],
		 [  0,   0,   0,   0],
		 [  0,   0,   100,   0]],

		[[  0,   1,   0,   0],
		 [  0,   0,   0,   0],
		 [  0,   1,   0,   0],
		 [  0,   1,   0,   0]],

		[[  0,   0,   0,   0],
		 [  0,   2,   0,   0],
		 [  0,   0,   0,   0],
		 [  0,   0,   0,   0]]]
	)
	# print(scores, "input")
	# print(g, i, j, scores[g, i, j])
	return scores[g, i, j]

def optimize_spans(scores):
	# print(scores.shape)
	# print("scores: ", scores)
	n = len(scores)
	C = torch.zeros([n, n, n], requires_grad = True)
	I = torch.zeros([n, n, n])

	def update(g, i, j):
		# print(n, g, i, j)
		# print("scores: ", scores[g, i, j], "C: ", C[g, i, i])
		I[g, i, j] = torch.max(C[g, i, i: j] + C[i, j, i + 1: j + 1]).item() + scores[g, i, j].item()
		I[g, j, i] = torch.max(C[g, j, i + 1: j + 1] + C[j, i, i: j]).item() + scores[g, j, i].item()
		C[g, i, j] = torch.max(I[g, i, i + 1: j + 1] + C[i, i + 1: j + 1, j])
		C[g, j, i] = torch.max(I[g, j, i: j] + C[j, i: j, i])

	n = n - 1
	for w in range(1, n):
		for i in range(1, n - w + 1):
			j = i + w
			g_list = [g for g in range(0, n + 1) if g < i or g > j]
			# print("i", i, "j", j, "n", n, g_list)
			for g in g_list:
				# print("i", i, "w", w, "g", g, "j", j)
				update(g, i, j)
	# print(C)
	# scores = C[0]
	# print(C)
	# print(scores, ">>>>>>>>>>>>>")
	# print("C[0]", C[0], "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLOOOOOOOOOO")
	result = parser.parse_proj(C[0])
	# print(result, "KKKKKKKKKKKKKK")
	return result

# 1       Ms.     _       NNP     _       _       2       TITLE   _       _
# 2       Haag    _       NNP     _       _       3       DEP     _       _
# 3       plays   _       VBZ     _       _       0       ROOT    _       _
# 4       Elianti _       NNP     _       _       3       OBJ     _       _
# 5       .       _       .       _       _       3       P       _       _
# x = ["", "Ms.", "Haag", "plays"]
# print(optimize_spans(x), "..............")
