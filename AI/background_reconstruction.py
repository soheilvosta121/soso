import cv2
import numpy as np
import helpers.background_saver as bs

class BackgroundReconstruction:
    def __init__(self, blur_kernel=None, custom_kernel=None):
        self.background_dic = {}
        self.index_file_background = 0        
        self.backgroud_saver = bs.BackgroundSaver()
        self.backgroud_histoy = []
        self.frame_height = None
        self.frame_width = None

        self.scale_alg = cv2.INTER_AREA
        

        self.blur_kernel = blur_kernel if blur_kernel is not None else np.array([
            [0.11, 0.11, 0.11],
            [0.11, 0.11, 0.11],
            [0.11, 0.11, 0.11]
        ])
        self.custom_kernel = custom_kernel if custom_kernel is not None else np.array([
            [0, 0.1, 0],
            [0.1, 0.1, 0.1],
            [0, 0.1, 0]
        ])

        self.index = 0
        self.is_first_frame = False

        self.dim = None
        self.dim2 = None


        self.joined = None
        self.video_background = None
        self.frame_alpha = None
        self.frame_first_blur = None

    

    def background_builder(self, next_frame, boxes_to_exclude):
        if next_frame is None:
            return self.video_background
        
        frame = next_frame.copy()
        if self.index == 0:
            self.frame_height, self.frame_width, _chan = frame.shape
            self.dim = (self.frame_width, self.frame_height)
            self.dim2 = (self.frame_width // 4, self.frame_height // 4)

            self.joined = np.zeros((self.frame_height, self.frame_width, 1), dtype=np.uint8)
            self.video_background = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)
            self.frame_alpha = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)

        # Create a transparent RGBA mask
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8) #* 255  # Fully opaque mask

        for bbox in boxes_to_exclude:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  
            #cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0, 0), -1)  # Transparent (alpha = 0) for excluded areas
            print(f"Excluding Bounding Box: {bbox}")


        frame_s = frame.copy()
        self.index += 1

        if not self.is_first_frame:
            self.frame_first = frame_s
            self.frame_first_blur = cv2.filter2D(self.frame_first, -1, self.blur_kernel)
            self.is_first_frame = True
            return self.video_background

        frame_blur = cv2.filter2D(frame_s, -1, self.blur_kernel)
        diff_frame = cv2.absdiff(frame_blur, self.frame_first_blur)

        frame_gray = cv2.cvtColor(diff_frame, cv2.COLOR_RGB2GRAY)
        _, bin_frame = cv2.threshold(frame_gray, 30, 255, cv2.THRESH_BINARY)

        self.joined = cv2.bitwise_or(self.joined, bin_frame)
        self.joined = cv2.bitwise_or(self.joined, mask)

        # TODO: Make 100 frames configurable
        if self.index % 100 == 0:
            self.is_first_frame = False
		    
            # Filtering

            #self.frame_first = frame_s
            #self.frame_first_blur = cv2.filter2D(self.frame_first, -1, self.blur_kernel)
            frame_s1 = cv2.resize(self.joined, self.dim2, interpolation=cv2.INTER_LINEAR)
            filtered_image = cv2.filter2D(frame_s1, -1, self.custom_kernel)
            _, frame_s1 = cv2.threshold(filtered_image, 35, 255, cv2.THRESH_BINARY)
            self.joined = cv2.resize(frame_s1, self.dim, interpolation=cv2.INTER_NEAREST)
            self.joined = cv2.filter2D(self.joined, -1, self.blur_kernel)

            self.frame_alpha[:, :, :3] = self.frame_first
            self.frame_alpha[:, :, 3] = 255 - self.joined

            alpha_background = self.video_background[:, :, 3] / 255.0
            alpha_overlay = self.frame_alpha[:, :, 3] / 255.0

            for color in range(3):
                self.video_background[:, :, color] = (
                    alpha_overlay * self.frame_alpha[:, :, color]
                    + alpha_background * self.video_background[:, :, color] * (1 - alpha_overlay)
                )

            self.video_background[:, :, 3] = (1 - (1 - alpha_overlay) * (1 - alpha_background)) * 255
            self.backgroud_histoy.append(self.video_background.copy())
            
            
            #self.background_dic[self.index_file_background] = self.video_background

            #cv2.imwrite(f"background_{self.index_file_background}.png", self.video_background)
            self.index_file_background += 1
            self.joined.fill(0)
        return self.video_background

    def release(self):
        """Release video resources."""
        self.cap.release()
        cv2.destroyAllWindows()


    def divide_image(self, image, num_vertical, num_horizontal):
    
        if num_vertical <= 0 or num_horizontal <= 0:
            raise ValueError("Number of divisions must be positive integers.")

        height, width = image.shape[:2]  # Get image dimensions (handles color or grayscale)

        # Calculate sub-image size (integer division, effectively cropping if not divisible)
        sub_height = height // num_vertical
        sub_width = width // num_horizontal

        if sub_height == 0 or sub_width == 0:
            raise ValueError("Division results in zero-sized sub-images. Reduce the number of divisions.")

        # Warn if cropping is needed
        if height % num_vertical != 0 or width % num_horizontal != 0:
            print(f"Warning: Image dimensions ({height}x{width}) not perfectly divisible by {num_vertical}x{num_horizontal}. "
                  f"Cropping to ({sub_height * num_vertical}x{sub_width * num_horizontal}).")

        sub_images = []

        for i in range(num_vertical):
            for j in range(num_horizontal):
                # Slice the sub-image
                sub_image = image[i * sub_height:(i + 1) * sub_height, j * sub_width:(j + 1) * sub_width]
                sub_images.append(sub_image)

        print(f"Total sub-images created for this image: {len(sub_images)}")
        return sub_images
