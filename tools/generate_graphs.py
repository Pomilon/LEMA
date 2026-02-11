import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('ggplot')
COLORS = {'PEFT': '#e74c3c', 'LEMA': '#2ecc71'}

def generate_vram_graph():
    models = ['GPT2\n(124M)', 'TinyLlama\n(1.1B)', 'SmolLM2\n(1.7B)', 'Llama-2\n(7B)']
    peft_vram = [0.44, 2.67, 3.88, 13.99] # 7B is Load-only, OOM on train
    lema_vram = [1.05, 2.12, 3.20, 5.90]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, peft_vram, width, label='Standard PEFT', color=COLORS['PEFT'])
    rects2 = ax.bar(x + width/2, lema_vram, width, label='LEMA', color=COLORS['LEMA'])
    
    # Add labels
    ax.set_ylabel('Peak VRAM Usage (GB)')
    ax.set_title('VRAM Usage: Standard PEFT vs LEMA\n(Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend() 
    
    # Add OOM annotation for 7B PEFT
    # rects1[3] is Llama-7B PEFT bar
    ax.text(rects1[3].get_x() + rects1[3].get_width()/2., 14.5,
            'OOM on Train',
            ha='center', va='bottom', color='red', fontweight='bold')

    # Add Savings labels
    def autolabel(rects, savings=None):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{height:.1f}GB', 
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            if savings and savings[i] is not None:
                 ax.annotate(f'{savings[i]}', 
                        xy=(rect.get_x() + rect.get_width() / 2, height/2),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center', color='white', fontweight='bold', rotation=90)

    autolabel(rects1)
    autolabel(rects2, [None, '-20%', '-17%', '-58%'])
    
    plt.tight_layout()
    plt.savefig('docs/assets/vram_benchmark.png')
    print("Generated docs/assets/vram_benchmark.png")

def generate_speed_graph():
    # Only Llama models for speed test
    models = ['TinyLlama\n(1.1B)', 'Llama-2\n(7B)']
    peft_time = [0.46, 0] # 0 for OOM
    lema_time = [1.45, 7.21]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, peft_time, width, label='Standard PEFT', color=COLORS['PEFT'])
    rects2 = ax.bar(x + width/2, lema_time, width, label='LEMA', color=COLORS['LEMA'])
    
    ax.set_ylabel('Time per Step (Seconds)')
    ax.set_title('Training Speed: Standard PEFT vs LEMA')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    # Annotate OOM for 7B PEFT
    ax.text(rects1[1].get_x() + rects1[1].get_width()/2., 0.5,
            'FAILED\n(OOM)',
            ha='center', va='bottom', color='red', fontweight='bold')
    
    # Annotate Overhead for 1.1B
    height_peft = peft_time[0]
    height_lema = lema_time[0]
    ax.annotate(f'{height_peft:.2f}s', 
                xy=(rects1[0].get_x() + rects1[0].get_width() / 2, height_peft),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    ax.annotate(f'{height_lema:.2f}s', 
                xy=(rects2[0].get_x() + rects2[0].get_width() / 2, height_lema),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    ax.text(x[0], height_lema + 0.2, f"~3.1x Slower\n(Latency Overhead)", ha='center')
    
    # Annotate 7B LEMA
    height_7b = lema_time[1]
    ax.annotate(f'{height_7b:.2f}s', 
                xy=(rects2[1].get_x() + rects2[1].get_width() / 2, height_7b),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    ax.text(rects2[1].get_x() + rects2[1].get_width()/2, height_7b + 0.5, "SUCCESS", ha='center', color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('docs/assets/speed_benchmark.png')
    print("Generated docs/assets/speed_benchmark.png")

if __name__ == "__main__":
    generate_vram_graph()
    generate_speed_graph()
