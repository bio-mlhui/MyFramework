

def generate_windows_of_video_v0(all_frames, 
                                    window_size,
                                    window_step, 
                                    force_not_interleave=None, force_all_used=None,):
    all_frames = sorted(all_frames)
    if window_size is None:
        assert window_step == None
        sampled_windows = [all_frames]
        return sampled_windows
    else:
        if force_not_interleave:
            assert window_step >= window_size
        if force_all_used:
            assert window_step <= window_size

        sampled_windows = []
        for i in range(0, len(all_frames), window_step):
            sampled_windows.append(all_frames[i:i+window_size])
            if i + window_size >= (len(all_frames)-1):
                # 第一次超过
                break
        if force_not_interleave and force_all_used:
            assert sum([len(win) for win in sampled_windows]) == len(all_frames)
    
        return sampled_windows

    
def generate_windows_of_video(all_frames, window_size, window_step, 
                              force_not_interleave, force_all_used,
                              pad_last_window=False):
    all_frames = sorted(all_frames)
    if window_size is None:
        assert window_step == None
        sampled_windows = [all_frames]
        return sampled_windows
    else:
        if force_not_interleave:
            assert not pad_last_window
            assert window_step >= window_size
        if force_all_used:
            assert window_step <= window_size

        sampled_windows = []
        for i in range(0, len(all_frames), window_step):
            sampled_windows.append(all_frames[i:(i+window_size)])

        if force_not_interleave and force_all_used:
            assert sum([len(win) for win in sampled_windows]) == len(all_frames)

        if pad_last_window:
            last_window_len = len(sampled_windows[-1])
            if last_window_len < window_size:
                if len(all_frames) > window_size:
                    sampled_windows[-1] = all_frames[-window_size:]
                else:
                    delta = window_size - last_window_len
                    sampled_windows[-1] = sampled_windows[-1] + [all_frames[-1]] * delta
            for wd in sampled_windows:
                assert len(wd) == window_size
        return sampled_windows
