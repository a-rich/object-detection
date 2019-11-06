import matplotlib.pyplot as plt
import pickle

data = pickle.load(open('results.pkl', 'rb'))
tiny = ('tiny (inf)', 1/data['yolo_0.6_2_720p.avi']['inference_fps'])
full = ('full (inf)', 1/data['full_yolo_0.6_2_720p.avi']['inference_fps'])
tiny2 = ('tiny (eff)', data['yolo_0.6_2_720p.avi']['effective_fps'])
full2 = ('full (eff)', data['full_yolo_0.6_2_720p.avi']['effective_fps'])

plt.bar(range(4), [tiny[1], tiny2[1], full[1], full2[1]], tick_label=[tiny[0],
    tiny2[0], full[0], full2[0]])
#plt.legend('upper right')
plt.show()
