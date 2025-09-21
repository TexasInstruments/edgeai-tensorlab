# Copyright (c) 2018-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

##############################################################################

# Also includes parts from: https://github.com/open-mmlab/mmpose
# License: https://github.com/open-mmlab/mmpose/blob/master/LICENSE
#
# Copyright 2018-2020 Open-MMLab. All rights reserved.
#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright 2018-2020 Open-MMLab.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

##############################################################################

import numpy as np
from PIL import Image
from . import functional as F

import math
import copy

from scipy.spatial import transform
from pyquaternion import Quaternion
from ..utils.config_utils.misc_utils import inverse_sigmoid as inverse_sigmoid

_pseudo_referernce_points_128 = np.array(
       [[0.7141, 0.0779, 0.3798],
        [0.6558, 0.2019, 0.1328],
        [0.5498, 0.0753, 0.2593],
        [0.4186, 0.6960, 0.6485],
        [0.7797, 0.9463, 0.6834],
        [0.3145, 0.4969, 0.1813],
        [0.6174, 0.2292, 0.0715],
        [0.0545, 0.8516, 0.0346],
        [0.8019, 0.3775, 0.5654],
        [0.2236, 0.2272, 0.0065],
        [0.8640, 0.4546, 0.5294],
        [0.5584, 0.9719, 0.1916],
        [0.3135, 0.7178, 0.2736],
        [0.5393, 0.3042, 0.8716],
        [0.7016, 0.8912, 0.7680],
        [0.4847, 0.2798, 0.6634],
        [0.7753, 0.5995, 0.7540],
        [0.7451, 0.6262, 0.3553],
        [0.2052, 0.2140, 0.8312],
        [0.3272, 0.9784, 0.1980],
        [0.5477, 0.5155, 0.3641],
        [0.6599, 0.3679, 0.4826],
        [0.5877, 0.1345, 0.6798],
        [0.6893, 0.6642, 0.6826],
        [0.6985, 0.2303, 0.7425],
        [0.0718, 0.5600, 0.4242],
        [0.8099, 0.7495, 0.0543],
        [0.1684, 0.1660, 0.8126],
        [0.3485, 0.3117, 0.6042],
        [0.9822, 0.5864, 0.7306],
        [0.8979, 0.7930, 0.5774],
        [0.1346, 0.1851, 0.5194],
        [0.8768, 0.9565, 0.4170],
        [0.5636, 0.4484, 0.8091],
        [0.7291, 0.4814, 0.3839],
        [0.3434, 0.2149, 0.8171],
        [0.2034, 0.2946, 0.0936],
        [0.9422, 0.1984, 0.7616],
        [0.3120, 0.3645, 0.9267],
        [0.1693, 0.9390, 0.2286],
        [0.6217, 0.8452, 0.0096],
        [0.6940, 0.5485, 0.2000],
        [0.2264, 0.5925, 0.6678],
        [0.4798, 0.3771, 0.7166],
        [0.2900, 0.5393, 0.3652],
        [0.4753, 0.5967, 0.1383],
        [0.4909, 0.6563, 0.7579],
        [0.8413, 0.6262, 0.6284],
        [0.0590, 0.0275, 0.5858],
        [0.0190, 0.8125, 0.1043],
        [0.7902, 0.2197, 0.4518],
        [0.8513, 0.9528, 0.2141],
        [0.8212, 0.6302, 0.3201],
        [0.1996, 0.6273, 0.6570],
        [0.7245, 0.8263, 0.1150],
        [0.2950, 0.5964, 0.6718],
        [0.0695, 0.8401, 0.5742],
        [0.4051, 0.1942, 0.1136],
        [0.7377, 0.7822, 0.7816],
        [0.4058, 0.0481, 0.8807],
        [0.2701, 0.4483, 0.0992],
        [0.0032, 0.1512, 0.0800],
        [0.5537, 0.2269, 0.0065],
        [0.8844, 0.5902, 0.2775],
        [0.4935, 0.6983, 0.3673],
        [0.2974, 0.1511, 0.7244],
        [0.2613, 0.9759, 0.5784],
        [0.5921, 0.0170, 0.6403],
        [0.6045, 0.6795, 0.3722],
        [0.0081, 0.9833, 0.8263],
        [0.9634, 0.5345, 0.2606],
        [0.0035, 0.2321, 0.3219],
        [0.6939, 0.6974, 0.2989],
        [0.6737, 0.8648, 0.7250],
        [0.0886, 0.7380, 0.4014],
        [0.9756, 0.1075, 0.8245],
        [0.5762, 0.4630, 0.0812],
        [0.6084, 0.1922, 0.8952],
        [0.6058, 0.1171, 0.1737],
        [0.9764, 0.2783, 0.6157],
        [0.2439, 0.4426, 0.4483],
        [0.7261, 0.8861, 0.7062],
        [0.6705, 0.4358, 0.3852],
        [0.6367, 0.6146, 0.6860],
        [0.8662, 0.1860, 0.5927],
        [0.1593, 0.8006, 0.7105],
        [0.7641, 0.7599, 0.4194],
        [0.4399, 0.9076, 0.6091],
        [0.1556, 0.0279, 0.0222],
        [0.1333, 0.0180, 0.9065],
        [0.5844, 0.9775, 0.9615],
        [0.8292, 0.1813, 0.2349],
        [0.6957, 0.7578, 0.7125],
        [0.5098, 0.3085, 0.6060],
        [0.9763, 0.9699, 0.7969],
        [0.3433, 0.0459, 0.0690],
        [0.8582, 0.0899, 0.3280],
        [0.0545, 0.9454, 0.7869],
        [0.5824, 0.3765, 0.8862],
        [0.4608, 0.5933, 0.3052],
        [0.1210, 0.4022, 0.5403],
        [0.5762, 0.7996, 0.3703],
        [0.3889, 0.2901, 0.5118],
        [0.3478, 0.4663, 0.0786],
        [0.2531, 0.0998, 0.7974],
        [0.1780, 0.2529, 0.8225],
        [0.0783, 0.3224, 0.8154],
        [0.8215, 0.0321, 0.9043],
        [0.7793, 0.3411, 0.8342],
        [0.8379, 0.8804, 0.3857],
        [0.2141, 0.9988, 0.8447],
        [0.9308, 0.1014, 0.8317],
        [0.5000, 0.7720, 0.3387],
        [0.2736, 0.1193, 0.6800],
        [0.9349, 0.7011, 0.8907],
        [0.7938, 0.6652, 0.5292],
        [0.6591, 0.5632, 0.3513],
        [0.9234, 0.1804, 0.0102],
        [0.3617, 0.4843, 0.2803],
        [0.2391, 0.2577, 0.9714],
        [0.8813, 0.2402, 0.5143],
        [0.5917, 0.0953, 0.1079],
        [0.7877, 0.8140, 0.5015],
        [0.2725, 0.1663, 0.9106],
        [0.9360, 0.1157, 0.4712],
        [0.9335, 0.4651, 0.3197],
        [0.4271, 0.8732, 0.6994],
        [0.1346, 0.0948, 0.9743]])

_pseudo_referernce_points_256 = np.array(
       [[5.4999e-01, 4.8900e-01, 8.8253e-01],
        [7.2395e-01, 8.8065e-01, 5.0285e-01],
        [1.7710e-01, 2.8237e-01, 6.5893e-01],
        [5.2383e-01, 7.5380e-01, 5.4258e-01],
        [7.4490e-01, 5.0666e-01, 3.7661e-01],
        [9.4949e-01, 9.9939e-01, 8.3717e-02],
        [2.8084e-01, 3.7683e-01, 6.8835e-01],
        [7.3596e-01, 1.3249e-01, 5.4291e-01],
        [9.4871e-01, 5.9287e-02, 8.6421e-02],
        [8.0671e-01, 3.9840e-01, 8.1841e-01],
        [8.2771e-01, 1.7767e-01, 6.2883e-02],
        [2.8159e-02, 5.2333e-01, 7.5006e-01],
        [1.9769e-01, 4.8809e-01, 8.3212e-01],
        [6.6701e-01, 5.8703e-01, 8.1425e-01],
        [1.7640e-01, 6.6530e-01, 9.3726e-02],
        [6.0925e-01, 4.3926e-01, 6.0962e-01],
        [4.6939e-01, 4.1241e-01, 7.8501e-01],
        [7.7197e-01, 8.1122e-03, 6.1451e-01],
        [6.6579e-01, 7.3769e-01, 1.7488e-02],
        [6.3730e-01, 4.4505e-01, 5.7122e-01],
        [7.2170e-01, 3.3060e-01, 5.5368e-01],
        [4.5772e-01, 2.0616e-01, 4.9650e-01],
        [7.1188e-01, 9.8974e-01, 2.8198e-01],
        [5.9926e-01, 6.5255e-01, 5.9862e-02],
        [8.5615e-01, 9.1955e-01, 5.5674e-01],
        [9.3861e-01, 7.2174e-01, 9.2290e-01],
        [9.4183e-01, 8.7861e-01, 6.3699e-01],
        [9.9575e-01, 9.6485e-01, 2.6690e-01],
        [3.0033e-01, 5.9173e-01, 1.3612e-01],
        [7.6322e-01, 5.6971e-01, 7.3067e-01],
        [6.8701e-01, 7.4947e-01, 3.0744e-01],
        [4.3376e-01, 1.9726e-01, 5.9810e-01],
        [3.0469e-01, 8.1395e-01, 9.7898e-01],
        [3.0919e-01, 2.7211e-01, 9.3117e-01],
        [3.1709e-01, 4.5427e-01, 2.7580e-01],
        [8.4075e-01, 7.5259e-01, 6.2469e-01],
        [8.7341e-01, 1.2710e-01, 1.3857e-03],
        [5.4846e-01, 7.9017e-01, 6.8721e-02],
        [3.5740e-01, 4.8504e-01, 5.9127e-01],
        [5.1714e-01, 4.9310e-01, 2.3402e-01],
        [2.5355e-01, 8.7206e-01, 2.9314e-01],
        [9.3937e-02, 3.4868e-01, 4.8145e-01],
        [4.7807e-01, 1.5502e-01, 1.3230e-03],
        [9.3321e-02, 9.5126e-01, 5.8614e-01],
        [8.0163e-01, 5.5830e-01, 6.2478e-01],
        [1.6032e-01, 9.8884e-01, 5.9101e-01],
        [8.9919e-01, 8.9445e-01, 9.7925e-01],
        [9.8034e-01, 1.2918e-01, 7.9747e-01],
        [1.3559e-01, 4.1321e-01, 3.5743e-01],
        [6.5989e-01, 1.3324e-01, 9.4177e-01],
        [2.4566e-01, 6.6357e-01, 1.6739e-01],
        [6.5011e-01, 6.9804e-01, 5.8802e-01],
        [6.7510e-01, 1.1578e-01, 7.6513e-01],
        [9.2514e-01, 2.4161e-01, 9.5959e-01],
        [9.8394e-01, 5.2368e-01, 4.2420e-01],
        [5.0858e-01, 4.4350e-01, 2.7148e-01],
        [5.4009e-01, 2.0972e-01, 5.6967e-01],
        [7.6555e-01, 4.4116e-01, 6.5278e-01],
        [8.1958e-01, 5.3398e-01, 6.0505e-01],
        [5.4721e-01, 8.8046e-01, 7.9567e-01],
        [3.2975e-01, 4.2307e-01, 4.3976e-01],
        [5.3389e-01, 3.2235e-01, 7.5983e-01],
        [2.2208e-01, 9.6234e-01, 5.1991e-01],
        [8.2613e-01, 2.5969e-01, 4.0298e-01],
        [7.0119e-01, 3.0308e-01, 1.4032e-02],
        [3.9394e-01, 5.7794e-01, 5.9244e-01],
        [8.6196e-01, 4.9775e-01, 9.6580e-01],
        [4.1694e-01, 8.7054e-01, 3.8906e-01],
        [2.5608e-01, 8.4696e-01, 6.5997e-01],
        [6.0534e-02, 1.0604e-01, 1.3401e-01],
        [7.4819e-01, 5.3630e-01, 8.2733e-01],
        [5.5966e-01, 9.1123e-01, 5.8197e-02],
        [9.7105e-01, 6.3208e-01, 1.1986e-02],
        [7.9068e-02, 4.6573e-01, 3.9377e-01],
        [9.9085e-01, 5.2066e-01, 1.7744e-02],
        [8.7206e-01, 4.1000e-01, 5.2882e-01],
        [6.2615e-01, 5.6934e-02, 9.0714e-02],
        [7.6010e-01, 1.1577e-01, 9.6826e-01],
        [2.7160e-01, 6.8428e-02, 6.7173e-01],
        [1.2088e-02, 1.9949e-01, 2.1752e-01],
        [5.1173e-01, 7.4123e-01, 5.1306e-01],
        [7.6247e-01, 7.2908e-01, 8.6810e-01],
        [2.4670e-01, 1.2980e-01, 4.4069e-01],
        [8.7477e-02, 7.3344e-01, 5.9714e-01],
        [8.7080e-01, 1.7475e-01, 9.1044e-01],
        [4.9332e-01, 3.5905e-02, 5.7622e-01],
        [7.7711e-01, 3.5533e-01, 7.5440e-01],
        [5.0714e-01, 8.9034e-01, 6.9835e-01],
        [3.9832e-01, 4.4964e-01, 8.7494e-01],
        [3.8338e-01, 5.2652e-01, 2.8646e-01],
        [3.4765e-01, 5.9484e-01, 2.5352e-01],
        [9.5306e-01, 6.1492e-01, 7.7629e-01],
        [7.8841e-01, 1.2542e-01, 1.3153e-02],
        [7.9762e-01, 3.2411e-01, 6.0791e-01],
        [7.0713e-01, 2.5107e-01, 8.6345e-02],
        [2.7195e-01, 7.8728e-01, 3.3866e-01],
        [2.1858e-01, 1.7288e-01, 6.4038e-01],
        [4.5295e-01, 4.3841e-01, 5.3554e-02],
        [5.8132e-01, 1.4092e-01, 7.0494e-03],
        [9.8400e-01, 4.7953e-01, 4.5781e-01],
        [8.3261e-01, 4.2656e-01, 4.9224e-01],
        [6.6133e-01, 2.8454e-01, 5.7535e-02],
        [2.4398e-01, 1.2939e-01, 3.3104e-01],
        [8.3097e-02, 3.3692e-01, 8.3049e-02],
        [7.9330e-01, 2.2559e-01, 4.8333e-01],
        [8.1885e-01, 5.5214e-02, 3.7557e-01],
        [2.0987e-01, 4.5451e-01, 9.1644e-01],
        [2.3133e-01, 5.2619e-01, 7.7155e-01],
        [4.8017e-01, 9.0453e-02, 1.7960e-01],
        [4.4492e-01, 7.3908e-01, 2.8010e-01],
        [6.5758e-01, 2.0534e-01, 5.6694e-01],
        [2.4327e-01, 3.6112e-01, 5.2083e-01],
        [7.9944e-01, 4.8219e-01, 3.4746e-01],
        [8.3870e-01, 1.5212e-01, 3.3899e-01],
        [9.4856e-01, 9.9385e-01, 1.8849e-01],
        [9.5115e-01, 7.2035e-01, 9.1604e-01],
        [6.6035e-01, 3.0698e-01, 2.0320e-01],
        [5.1230e-01, 1.9482e-01, 4.4915e-02],
        [9.0055e-01, 1.4569e-01, 4.4924e-01],
        [7.3917e-01, 7.8169e-01, 3.7758e-01],
        [4.7099e-01, 3.3240e-01, 7.8008e-02],
        [7.6191e-01, 1.1984e-01, 6.4123e-01],
        [2.7276e-01, 9.1517e-01, 6.9994e-01],
        [2.2148e-01, 3.1175e-01, 5.9494e-01],
        [3.9756e-01, 7.5182e-01, 4.9843e-01],
        [8.1775e-01, 7.8403e-01, 5.8615e-01],
        [9.4749e-01, 3.2128e-01, 3.2608e-01],
        [2.1096e-01, 5.8491e-01, 9.9148e-02],
        [5.2482e-01, 6.7415e-01, 6.9652e-01],
        [9.0139e-01, 8.9660e-01, 3.1184e-01],
        [6.5434e-01, 6.9435e-01, 5.5979e-01],
        [8.5453e-01, 8.0218e-01, 2.7269e-01],
        [7.8144e-01, 6.9701e-01, 1.9905e-01],
        [2.3476e-01, 8.2989e-01, 9.9212e-01],
        [9.2143e-01, 1.4510e-02, 8.1253e-01],
        [4.2563e-02, 2.9787e-01, 6.9955e-01],
        [5.7895e-01, 8.0899e-02, 4.5980e-01],
        [7.3353e-02, 6.7184e-02, 8.0107e-01],
        [3.0364e-01, 4.6660e-01, 4.3550e-01],
        [9.2832e-01, 9.6250e-01, 6.8170e-01],
        [8.6343e-01, 6.1228e-02, 2.9699e-01],
        [3.2573e-01, 9.3183e-02, 2.9373e-01],
        [9.9350e-02, 4.2402e-01, 4.0638e-01],
        [2.9338e-01, 2.7073e-01, 6.8156e-01],
        [2.3349e-01, 9.2699e-01, 5.6365e-01],
        [9.7807e-01, 8.9649e-01, 7.7572e-01],
        [3.8410e-01, 9.8068e-01, 1.1426e-01],
        [9.2098e-01, 3.7048e-01, 9.8193e-01],
        [4.1715e-01, 8.0062e-01, 5.8439e-01],
        [5.2705e-02, 1.8374e-01, 6.6237e-01],
        [2.0310e-01, 4.6950e-01, 2.4938e-01],
        [7.6177e-01, 6.8487e-01, 8.5405e-01],
        [1.5041e-02, 6.0606e-01, 7.7205e-01],
        [3.8232e-01, 2.6535e-02, 3.5767e-01],
        [8.2541e-02, 2.1282e-02, 2.5894e-01],
        [3.4832e-01, 9.8243e-05, 7.5366e-01],
        [7.2037e-01, 2.2624e-01, 1.2242e-02],
        [1.5937e-01, 5.0127e-02, 1.0687e-01],
        [3.3306e-01, 8.2125e-01, 6.7234e-01],
        [6.6317e-01, 4.9151e-01, 4.9424e-01],
        [8.9123e-01, 7.6363e-01, 4.9323e-01],
        [7.0688e-01, 9.1364e-01, 2.9037e-01],
        [6.8809e-01, 9.9510e-01, 8.7019e-01],
        [7.3723e-01, 7.4401e-01, 9.7590e-02],
        [2.2528e-01, 3.9140e-02, 2.1468e-01],
        [6.1155e-01, 2.3840e-01, 9.4987e-01],
        [6.1736e-01, 9.4040e-01, 1.6660e-02],
        [1.0579e-01, 9.9970e-01, 6.2291e-01],
        [6.8248e-01, 1.1132e-01, 5.7771e-01],
        [9.1089e-01, 3.4259e-01, 7.8425e-01],
        [1.2253e-01, 4.6548e-01, 2.5246e-01],
        [4.9771e-01, 3.4553e-01, 7.6820e-01],
        [4.1780e-01, 2.2437e-01, 9.9830e-01],
        [5.3996e-01, 7.3248e-01, 7.4186e-01],
        [7.3189e-01, 2.1620e-01, 7.4850e-01],
        [5.5276e-01, 1.7857e-01, 6.6787e-01],
        [5.4463e-01, 6.2757e-01, 3.3788e-01],
        [4.1856e-01, 3.9239e-01, 9.6480e-01],
        [8.5407e-01, 2.8041e-01, 5.8303e-02],
        [1.4868e-01, 5.8628e-01, 5.2901e-01],
        [9.7082e-01, 1.5488e-01, 6.9846e-01],
        [6.9137e-01, 4.7125e-01, 6.0907e-01],
        [8.4272e-01, 7.8106e-01, 9.3954e-01],
        [9.0626e-01, 1.7219e-01, 3.7626e-01],
        [8.6868e-01, 2.9951e-02, 6.8613e-01],
        [3.2653e-01, 6.1730e-01, 8.0631e-01],
        [7.2760e-01, 5.4295e-01, 9.8084e-01],
        [2.0345e-02, 3.5636e-01, 6.3370e-01],
        [4.1283e-02, 1.0898e-01, 2.1717e-01],
        [1.1450e-01, 6.8004e-01, 8.0163e-01],
        [6.9081e-01, 9.1311e-01, 3.1224e-03],
        [1.4607e-01, 8.2656e-01, 6.3981e-01],
        [3.0315e-01, 2.7086e-01, 7.0724e-01],
        [1.9537e-01, 2.4014e-01, 1.1546e-01],
        [8.9559e-01, 6.6684e-02, 4.8682e-01],
        [7.9234e-01, 2.9550e-01, 6.7562e-01],
        [3.1669e-01, 1.9082e-03, 4.5749e-02],
        [4.1410e-01, 6.3522e-01, 5.8921e-01],
        [1.9322e-01, 4.7684e-01, 5.7249e-01],
        [4.9151e-01, 9.6998e-01, 3.4538e-01],
        [2.6812e-01, 9.2782e-01, 1.7657e-01],
        [6.8112e-01, 4.5518e-02, 3.2384e-01],
        [6.6184e-02, 1.3286e-01, 2.0985e-01],
        [4.3641e-01, 8.3208e-02, 5.2225e-01],
        [8.2285e-01, 7.4051e-01, 8.0263e-01],
        [4.2321e-01, 3.9832e-01, 2.3899e-01],
        [7.2801e-01, 4.5597e-02, 8.2590e-01],
        [9.0304e-01, 4.8984e-01, 6.6463e-01],
        [6.0301e-02, 9.1031e-01, 9.5373e-01],
        [5.5440e-01, 7.4098e-01, 9.7927e-01],
        [1.2017e-02, 5.8974e-01, 1.7011e-02],
        [9.2733e-01, 3.4443e-02, 2.8400e-01],
        [1.5944e-01, 3.3729e-01, 2.6240e-01],
        [9.1072e-02, 2.7163e-01, 5.4535e-01],
        [7.4748e-02, 4.6959e-01, 6.8685e-01],
        [4.4953e-01, 9.3854e-01, 1.4848e-01],
        [3.8079e-01, 6.8679e-01, 1.7821e-01],
        [1.5890e-01, 9.7207e-01, 4.3445e-01],
        [2.6469e-01, 1.4060e-01, 8.2211e-01],
        [4.7027e-01, 8.0659e-02, 5.6438e-01],
        [8.1444e-01, 3.4541e-01, 1.6126e-01],
        [7.9610e-02, 2.7958e-01, 6.1027e-01],
        [7.4312e-01, 3.5235e-01, 6.3902e-01],
        [6.5041e-01, 7.6661e-01, 7.3251e-01],
        [7.1580e-01, 3.4596e-01, 5.8912e-01],
        [8.5555e-02, 9.3284e-02, 8.6296e-01],
        [7.1013e-01, 8.3471e-01, 2.2300e-01],
        [2.9325e-01, 9.3512e-01, 5.7096e-01],
        [9.6507e-01, 9.6095e-01, 6.3784e-01],
        [5.3977e-01, 7.6330e-01, 6.7692e-02],
        [3.3106e-02, 9.4860e-01, 8.8756e-01],
        [5.9408e-01, 9.6724e-01, 7.7894e-01],
        [7.6588e-01, 2.4948e-01, 8.6008e-01],
        [2.8248e-01, 7.8676e-03, 6.5575e-01],
        [2.2677e-01, 7.2857e-01, 8.0056e-01],
        [6.3405e-01, 6.0117e-01, 4.1169e-02],
        [5.1235e-02, 6.1680e-01, 2.7021e-01],
        [2.4479e-01, 6.0936e-03, 6.0949e-01],
        [2.7631e-02, 8.7389e-01, 9.5041e-01],
        [9.1530e-01, 5.9385e-01, 5.2465e-01],
        [2.7286e-01, 1.9875e-01, 9.8019e-01],
        [5.8318e-01, 8.7934e-01, 9.4103e-01],
        [8.2265e-01, 6.8212e-01, 9.4216e-01],
        [1.0789e-01, 4.8848e-01, 7.6410e-01],
        [9.1581e-01, 2.3372e-01, 8.7306e-01],
        [5.8937e-01, 2.8460e-02, 6.5788e-01],
        [1.8843e-01, 9.7427e-01, 7.0697e-01],
        [8.1482e-01, 4.6571e-01, 6.1911e-01],
        [2.9462e-01, 4.2262e-01, 1.8539e-01],
        [3.8909e-01, 6.1497e-01, 2.6681e-01],
        [8.5674e-01, 8.8591e-01, 4.3712e-01],
        [6.7114e-01, 5.5456e-01, 2.5684e-01],
        [4.8306e-01, 3.5900e-01, 8.3604e-01],
        [2.4140e-01, 9.0728e-01, 2.5386e-01],
        [6.1764e-01, 1.4861e-01, 7.7350e-01],
        [6.2358e-01, 7.2064e-01, 9.7005e-01]])

# pulled exactly from the nuscenes official code for installation on evm
# from nuscenes.eval.common.utils import quaternion_yaw 
def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


_camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
]

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.reshape(-1, *view_shape[1:])
    return (memory * prev_exist).astype(memory.dtype)

def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = np.concatenate([reference_points, np.ones_like(reference_points[..., 0:1])], axis=-1)
    if reverse:
        matrix = np.linalg.inv(egopose)
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = np.squeeze(np.expand_dims(matrix, 1) @ np.expand_dims(reference_points, -1), -1)[..., :3]
    return reference_points


class ImageCrop():
    """Crops the given image at the specified region
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired area to crop. 
    """

    def __init__(self, dim=None):
        super().__init__()
        if dim is None:
            self.dim = dim
        else:
            if len(dim) != 4:
                raise ValueError("Please provide (top, left, height, width) for cropping area.")
            self.dim = dim
        #

    def __call__(self, img, info_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.dim is not None:
            if isinstance(img, list):
                for i in range(len(img)):
                    img[i] = F.crop(img[i], self.dim[1], self.dim[0], self.dim[3], self.dim[2])
            else:
                img = F.crop(img, self.dim[1], self.dim[0], self.dim[3], self.dim[2])

        return img, info_dict

    def __repr__(self):
        if self.dim is not None:
            return self.__class__.__name__ + '(dim={0})'.format(self.dim)
        else:
            return self.__class__.__name__ + '()'


class ImagePad():
    """Pad the given image at the specified paded image size
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired size to pad. 
    """

    def __init__(self, dim=None):#, pad_color=(103.530, 116.280, 123.675)):
        super().__init__()
        if dim is None:
            self.dim = dim
        else:
            if len(dim) != 4:
                raise ValueError("Please provide (left, top, right, bottom) to pad")
            self.dim = dim
        #self.pad_color = pad_color

    def __call__(self, img, info_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        if self.dim is not None:
            if isinstance(img, list):
                for i in range(len(img)):
                    img[i] = F.pad(img[i], self.dim)
                info_dict['pad_shape'] = (img[0].shape[0], img[0].shape[1])
            else:
                img = F.pad(img, self.dim)

                #img = img.astype(np.float32)
                ## left and top
                #if self.dim[0] != 0 or self.dim[1] !=0:
                #    img[:self.dim[1], :self.dim[0], 0] = self.pad_color[0]
                #    img[:self.dim[1], :self.dim[0], 1] = self.pad_color[1]
                #    img[:self.dim[1], :self.dim[0], 2] = self.pad_color[2]
                ## right and bottom
                #if self.dim[2] != 0 or self.dim[3] !=0:
                #    img[-self.dim[3]:, -self.dim[2]:, 0] = self.pad_color[0]
                #    img[-self.dim[3]:, -self.dim[2]:, 1] = self.pad_color[1]
                #    img[-self.dim[3]:, -self.dim[2]:, 2] = self.pad_color[2]
                info_dict['pad_shape'] = (img.shape[0], img.shape[1])
        return img, info_dict

    def __repr__(self):
        if self.dim is not None:
            return self.__class__.__name__ + '(dim={0})'.format(self.dim)
        else:
            return self.__class__.__name__ + '()'



class BEVSensorsRead():

    """ Set sensor (image) sizes for BEV network, and 
        initialize and update camera intrinsic/extrinsic parameters.

        Args:
            imsize, Tensor: source image size.
            resize, Tensor: image size after resizing iamge.
            crop, Tensor: croped area of resized image. It coulde be larger than 
                          resized image when resized image should be padded.
            load type, str: 'frame_based' for BEV, 'mv_image_based' for a single frame detection

        Returns:
            Sensor (image) file name list.
            Dictionary (info_dict) that has all necessary parameters.

    """
    def __init__(self, imsize, resize, crop, load_type='frame_based'):
        self.load_type = load_type
        self.camera_types = _camera_types
        self.imsize = imsize
        self.resize = resize
        self.crop = crop

        self.prev_frame_info = {
            'prev_bev_exist': False,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def get_sensor_transforms(self, data, cam_name):
        w, x, y, z = data['cams'][cam_name]['sensor2ego_rotation']
        
        # sweep sensor to sweep ego
        sensor2ego_rot = np.array(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = np.array(
            data['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = np.zeros((4,4))  # sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        
        # sweep ego to global
        w, x, y, z = data['ego2global_rotation']
        ego2global_rot = np.array(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = np.array(data['ego2global_translation'])
        ego2global = np.zeros((4,4)) # ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran

        return sensor2ego, ego2global


    def get_calib_data(self, data, info_dict):
        # For camera transforms
        intrins = []
        post_intrins = []
        sensor2egos = []
        ego2globals = []
        post_rots = []
        post_trans = []
        lidar2cams = []
        cam2lidars = []
        lidar2imgs = []
        lidar2imgs_org = []
        ego2imgs = []
        
        for cam_name,dic in data['cams'].items():
            intrin    = np.array(dic['cam_intrinsic']).astype(np.float32)
            lidar2cam = np.array(dic['lidar2sensor']).astype(np.float32)
            cam2lidar = np.linalg.inv(lidar2cam)

            sensor2ego, ego2global = \
                self.get_sensor_transforms(data, cam_name)
            
            post_rot = np.eye(3)
            post_tran = np.zeros(3)

            # imgsize is from info_dict
            scale = max(self.resize[0] / self.imsize[0], self.resize[1] / self.imsize[1])
            post_rot[:2, :2] *= scale
            post_tran[:2] -= np.array(self.crop[:2])

            # camera instrinsic after resizing and cropping
            temp = copy.deepcopy(post_rot)
            temp[:2, 2] = post_tran[:2]
            post_intrin = (temp @ intrin).astype(np.float32)

            # lidar2img transform after resizing and cropping
            lidar2img = np.eye(4)
            lidar2img[:3, :3] = post_intrin
            post_intrin = np.copy(lidar2img)
            lidar2img = (lidar2img @ lidar2cam).astype(np.float32)

            # lidar2img_org before resizing and cropping
            # Needed for visualization
            lidar2img_org = np.eye(4)
            lidar2img_org[:3, :3] = intrin
            # lidar2img_org = (lidar2img_org @ lidar2cam).astype(np.float32)

            # ego2img
            # Needed for visualization of BEVDet
            ego2img = np.eye(4)
            ego2img[:3, :3] = intrin
            intrin = np.copy(a=lidar2img_org)
            
            lidar2img_org = (lidar2img_org @ lidar2cam).astype(np.float32)
            ego2img = (ego2img @ np.linalg.inv(sensor2ego)).astype(np.float32)

            intrins.append(intrin)
            post_intrins.append(post_intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            lidar2cams.append(lidar2cam)
            cam2lidars.append(cam2lidar)
            lidar2imgs.append(lidar2img)
            lidar2imgs_org.append(lidar2img_org)
            ego2imgs.append(ego2img)

        # For BEV transform
        bda_mat = np.zeros((4, 4))
        bda_mat[0, 0] = bda_mat[1, 1] = bda_mat[2, 2] = bda_mat[3, 3] = 1
        
        # expand array assuming batch_size = 1
        info_dict['num_cams']       = len(data['cams'])
        info_dict['intrins']        = np.stack(intrins)
        info_dict['post_intrins']   = np.stack(post_intrins)
        info_dict['sensor2egos']    = np.expand_dims(np.stack(sensor2egos), 0)
        info_dict['ego2globals']    = np.expand_dims(np.stack(ego2globals), 0)
        info_dict['post_rots']      = np.expand_dims(np.stack(post_rots), 0)
        info_dict['post_trans']     = np.expand_dims(np.stack(post_trans), 0)
        info_dict['lidar2cams']     = lidar2cams # np.expand_dims(np.stack(lidar2cams), 0)
        info_dict['cam2lidars']     = cam2lidars
        info_dict['lidar2imgs']     = lidar2imgs # np.expand_dims(np.stack(lidar2imgs), 0)
        info_dict['lidar2imgs_org'] = lidar2imgs_org
        info_dict['ego2imgs']       = ego2imgs
        info_dict['bda']            = np.expand_dims(bda_mat, 0)
        info_dict['pad_shape']      = (self.crop[3], self.crop[2])
        info_dict['lidar2ego']      = np.array(data['lidar2ego'])
        info_dict['scene_token']    = data['scene_token']
        info_dict['timestamp']      = data['timestamp']
        info_dict['img_timestamp']  = [v['timestamp'] for k, v  in data['cams'].items()]
        info_dict['delta_timestamp']= [info_dict['timestamp'] - v['timestamp'] for k, v  in data['cams'].items()]
        info_dict['scene_token']    = data['scene_token']
        info_dict['timestamp']      = data['timestamp']
        info_dict['img_timestamp']  = [v['timestamp'] for k, v  in data['cams'].items()]
        info_dict['delta_timestamp']= [info_dict['timestamp'] - v['timestamp'] for k, v  in data['cams'].items()]

        if 'BEVFormer' in info_dict['task_name']:
            info_dict['prev_bev_exist'] = True
            if info_dict['scene_token'] != self.prev_frame_info['scene_token']:
                info_dict['prev_bev_exist'] = False

            # can_bus
            matrot = ego2globals[0]
            rotation = transform.Rotation.from_matrix(matrot[:3, :3]).as_quat()
            rotation = Quaternion(a=rotation[3], i=rotation[0], j=rotation[1], k=rotation[2])

            can_bus = data['can_bus']
            can_bus[:3] = matrot[:3, 3]
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle

            tmp_pos = copy.deepcopy(data['can_bus'][:3])
            tmp_angle = copy.deepcopy(data['can_bus'][-1])
            if info_dict['prev_bev_exist'] == True:
                can_bus[:3] -= self.prev_frame_info['prev_pos']
                can_bus[-1] -= self.prev_frame_info['prev_angle']
            else:
                can_bus[:3] = 0
                can_bus[-1] = 0

            info_dict['can_bus'] = can_bus

            self.prev_frame_info['scene_token'] = info_dict['scene_token']
            self.prev_frame_info['prev_pos']    = tmp_pos
            self.prev_frame_info['prev_angle']  = tmp_angle

        return info_dict

    def __call__(self, data, info_dict):
        if self.load_type == 'mv_image_based':
            info_dict['camera_type'] = data['camera_type']
            info_dict['ego2global']  = data['ego2global']
            info_dict['timestamp']   = data['timestamp']
            info_dict['intrins']     = data['images'][data['camera_type']]['cam_intrinsic']

            image_name = data['images'][data['camera_type']]['data_path']
            return image_name, info_dict
        else:
            image_name_list = []
            for cam, dic in data['cams'].items():
                image_name_list.append(dic['data_path'])

            # save lidar_path, which is also needed for visualization
            info_dict['lidar_path'] = data['lidar_path']
            info_dict = self.get_calib_data(data, info_dict)

            return tuple(image_name_list), info_dict


class GetPETRGeometry():
    def __init__(self, crop, featsize):
        # Params needed to generate coords3d: How make them configurable?
        # Batch size
        self.B = 1
        self.C              = 256
        self.H              = featsize[0]
        self.W              = featsize[1]
        self.feats_size = [1,6,256,20,50]

        self.position_level = 0
        self.with_multiview = True
        self.LID            = True
        self.depth_num      = 64
        self.depth_start    = 1
        self.position_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

    def add_lidar2img(self, img, meta):
        r"""add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        # for meta in batch_input_metas:
        img_shape = meta['data_shape'][:2]
        meta['img_shape'] = [img_shape] * len(meta['lidar2imgs'])

        return meta

    def prepare_data(self, img, info_dict):
        input_shape = img.shape[-2:]

        self.info_dict = info_dict

        # update real input shape of each single img
        # for img_meta in self.inp:
        info_dict.update(input_shape=input_shape)

    def create_coords3d(self, info_dict):
        batch_size = self.B
        #num_cams = info_dict['num_cams']
        num_cams = len(info_dict['lidar2imgs'])
        pad_h, pad_w = info_dict['pad_shape']

        # forward() in petr_head.py
        # masks is simply (B, N, self.H, self.W) array initialized to False
        masks = np.zeros((batch_size, num_cams, self.H, self.W)).astype(bool)

        eps = 1e-5
        B, N, C, H, W = self.B, num_cams, self.C, self.H, self.W
        coords_h = np.arange(H).astype(np.float32) * pad_h / H
        coords_w = np.arange(W).astype(np.float32) * pad_w / W

        if self.LID:
            index = np.arange(
                start=0,
                stop=self.depth_num,
                step=1).astype(np.float32)
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = np.arange(
                start=0,
                stop=self.depth_num,
                step=1).astype(np.float32)
            bin_size = (self.position_range[3] -
                        self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = np.transpose(np.stack(np.meshgrid(coords_w, coords_h, coords_d)), (2, 1, 3, 0))  # W, H, D, 3
        coords = np.concatenate((coords, np.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * np.maximum(
            coords[..., 2:3], np.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        img2lidar = []
        for i in range(len(info_dict['lidar2imgs'])):
            img2lidar.append(np.linalg.inv(info_dict['lidar2imgs'][i]))
        img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        
        #img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = np.repeat(coords.reshape(1, 1, W, H, D, 4, 1), N, 1)
        #img2lidars = img2lidars.reshape(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        img2lidars = np.repeat(img2lidars.reshape(B, N, 1, 1, 1, 4, 4), W, 2)
        img2lidars = np.repeat(img2lidars, H, 3)
        img2lidars = np.repeat(img2lidars, D, 4)
        coords3d = np.squeeze(np.matmul(img2lidars, coords), -1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        #coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = coords_mask.reshape(B, N, W, H, D*3).sum(-1) > (D*0.5)
        coords_mask = masks | coords_mask.transpose(0, 1, 3, 2)
        coords3d = np.ascontiguousarray(coords3d.transpose(0, 1, 4, 5, 3, 2)).reshape(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords3d = coords3d.astype(np.float32)

        return masks, coords3d

    def add_prev_img_metas(self, prev_img_meta, info_dict):
        e2g = np.array(info_dict['ego2globals'][0][0])
        l2e = np.array(info_dict['lidar2ego'])

        e2g_p = np.array(prev_img_meta['ego2globals'][0][0])
        l2e_p = np.array(prev_img_meta['lidar2ego'])

        prev_post_intrins = np.array(prev_img_meta['post_intrins'][0])

        for i in range(len(info_dict['lidar2cams'])):
            l2c_p = np.array(prev_img_meta['lidar2cams'][i])
            # Transform [R|t] from the (temporal) previous camera  to the current lidar
            cam2lidar_p_c = np.linalg.inv(l2e) @ np.linalg.inv(e2g) @ e2g_p @ l2e_p @ np.linalg.inv(l2c_p)
            # Transform [R|t] from the current lidar to the (temporal) previous camera
            lidar2cam_p_c = np.linalg.inv(cam2lidar_p_c)
            # Transform [R|t] from the current lidar to the (temporal) previous image
            post_intrins = np.eye(4)
            post_intrins[:3, :3] = prev_post_intrins[i]
            lidar2img_p_c = post_intrins @ lidar2cam_p_c

            # info_dict['post_intrins'].append(prev_post_intrins[i])
            info_dict['lidar2cams'].append(lidar2cam_p_c)
            # info_dict['lidar2cams'] = np.concatenate((, np.expand_dims(lidar2cam_p_c, axis=0)), axis=0)
            info_dict['lidar2imgs'].append(lidar2img_p_c)
            info_dict['delta_timestamp'].append(info_dict['timestamp'] - prev_img_meta['img_timestamp'][i])

        return info_dict

    def get_temporal_feats(self, info_dict):
        prev_feat = None
        prev_img_meta = info_dict
        prev_feats = []
        prev_img_metas = []

        if 'queue' in info_dict:
            num_prevs   = info_dict['num_bev_temporal_frames']
            queue_mem   = copy.deepcopy(info_dict['queue_mem'])
            feats_queue = info_dict['queue']
            del info_dict['queue_mem']
            del info_dict['queue']

            # Support only batch_size = 1s
            for i in range(1, num_prevs+1):
                cur_sample_idx = info_dict['sample_idx']

                if i > feats_queue.qsize() or \
                    info_dict['scene_token'] != queue_mem[cur_sample_idx - i]['img_meta']['scene_token']:
                    if prev_feat is None:
                        #prev_feats.append(np.zeros(self.feats_size, dtype=img.dtype))
                        prev_feats.append(np.zeros(self.feats_size, dtype=np.float32))
                        prev_img_metas.append(prev_img_meta)
                    else:
                        prev_feats.append(prev_feat)
                        prev_img_metas.append(prev_img_meta)
                else:
                    prev_feat = queue_mem[cur_sample_idx - i]['feature_map']
                    prev_img_meta = queue_mem[cur_sample_idx - i]['img_meta']
                    prev_feats.append(prev_feat)
                    prev_img_metas.append(prev_img_meta)
                self.add_prev_img_metas(prev_img_meta, info_dict)

        return np.concatenate(prev_feats, axis=0), prev_img_metas

    def __call__(self, data, info_dict):
        # get previous temporal infos
        prev_feats_map = None
        prev_input_metas = None

        ## for petr, combine all 6 images into on
        temp = np.concatenate(data, 0)
        data=[]
        data.append(temp)
        if 'num_bev_temporal_frames' in info_dict and info_dict['num_bev_temporal_frames'] > 0:
            prev_feats_map, prev_input_metas = self.get_temporal_feats(info_dict)
            info_dict = self.add_lidar2img(data, info_dict)
            self.prepare_data(data[0], info_dict)

        masks, coords3d = self.create_coords3d(info_dict)

        ## append coords3d. masks are not needed.
        data.append(coords3d)

        if 'num_bev_temporal_frames' in info_dict and info_dict['num_bev_temporal_frames'] > 0:
            if np.all(prev_feats_map == 0):
                valid_prev_feats= 0
            else:
                valid_prev_feats = 1
            data.append(np.array(valid_prev_feats, dtype=np.float32)) 
            data.append(prev_feats_map)

        return data, info_dict

class GetBEVDetGeometry():

    def __init__(self, crop):
        # how to configure these params?
        self.crop = crop
        self.downsample = 16
        self.grid_config = {
            'x': [-51.2, 51.2, 0.8],
            'y': [-51.2, 51.2, 0.8],
            'z': [-5, 3, 8],
            'depth': [1.0, 60.0, 1.0],
        }
        self.out_channels = 64

        self.create_grid_infos(**self.grid_config)

    def create_grid_infos(self, x, y, z, **kwargs):
        self.grid_lower_bound = np.array([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = np.array([cfg[2] for cfg in [x, y, z]])
        self.grid_size = np.array([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def create_frustum(self, info_dict):
        h_in, w_in = self.crop[3], self.crop[2]
        h_feat, w_feat = h_in // self.downsample, w_in // self.downsample

        depth_cfg = self.grid_config['depth']

        #d = np.arange(*depth_cfg, dtype=float) \
        #    .view(-1, 1, 1).expand(-1, h_feat, w_feat)
        d = np.arange(*depth_cfg, dtype=np.float32).reshape(-1, 1, 1)
        d = np.broadcast_to(d, (d.shape[0], h_feat, w_feat))
        depth_channels = d.shape[0]


        #x = np.linspace(0, w_in - 1, w_feat,  dtype=float)\
        #    .view(1, 1, w_feat).expand(depth_channels, h_feat, w_feat)
        #y = np.linspace(0, h_in - 1, h_feat,  dtype=float)\
        #    .view(1, h_feat, 1).expand(depth_channels, h_feat, w_feat)

        x = np.linspace(0, w_in - 1, w_feat, dtype=np.float32).reshape(1, 1, w_feat)
        x = np.broadcast_to(x, (depth_channels, h_feat, w_feat))
        y = np.linspace(0, h_in - 1, h_feat, dtype=np.float32).reshape(1, h_feat, 1)
        y = np.broadcast_to(y, (depth_channels, h_feat, w_feat))

        # D x H x W x 3
        return np.stack((x, y, d), -1)

    def get_lidar_coor(self, info_dict):

        cam2imgs    = info_dict['intrins']
        sensor2egos = info_dict['sensor2egos']
        post_rots   = info_dict['post_rots']
        post_trans  = info_dict['post_trans']
        bda         = info_dict['bda']

        frustum = self.create_frustum(info_dict)

        B, N, _, _ = sensor2egos.shape
        #N = sensor2egos.shape[0]

        # post-transformation
        # B x N x D x H x W x 3
        points = frustum - post_trans.reshape(B, N, 1, 1, 1, 3)
        #points = np.linalg.inv(post_rots).reshape(B, N, 1, 1, 1, 3, 3) \
        #    .matmul(points.unsqueeze(-1))
        points = np.matmul(np.linalg.inv(post_rots).reshape(B, N, 1, 1, 1, 3, 3), \
            np.expand_dims(points, -1))

        # cam_to_ego
        points = np.concatenate(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        #combine = sensor2egos[:,:,:3,:3].matmul(np.linalg.inv(cam2imgs))
        #points = combine.reshape(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        combine = np.matmul(sensor2egos[:,:,:3,:3], np.linalg.inv(cam2imgs))
        points = np.squeeze(np.matmul(combine.reshape(B, N, 1, 1, 1, 3, 3), points), -1)
        points += sensor2egos[:,:,:3, 3].reshape(B, N, 1, 1, 1, 3)
        #points = bda[:, :3, :3].reshape(B, 1, 1, 1, 1, 3, 3).matmul(
        #    points.unsqueeze(-1)).squeeze(-1)
        points = np.squeeze(np.matmul(bda[:, :3, :3].reshape(B, 1, 1, 1, 1, 3, 3), \
             np.expand_dims(points, -1)), -1)
        points += bda[:, :3, 3].reshape(B, 1, 1, 1, 1, 3)

        return points

    def precompute_voxel_info(self, coor):
        B, N, D, H, W, _ = coor.shape

        num_points = B * N * D * H * W

        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound) /
                self.grid_interval)

        coor = coor.astype(np.longlong).reshape(num_points, 3)
        #batch_idx = np.arange(0, B).reshape(B, 1). \
        #    expand(B, num_points // B).reshape(num_points, 1)
        batch_idx = np.arange(0, B).reshape(B, 1)
        batch_idx = np.broadcast_to(batch_idx, (B, num_points // B)).reshape(num_points, 1)
        coor = np.concatenate((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])

        # for our BEV pooling - coor in 1D tensor
        num_grids = (B*self.grid_size[2]*self.grid_size[1]*self.grid_size[0]).astype(np.int32)

        #bev_feat = np.zeros((num_grids + 1, self.out_channels), device=coor.device)
        bev_feat = np.zeros((num_grids + 1, self.out_channels), dtype=np.float32)

        coor_1d = np.zeros(num_points)
        coor_1d  = coor[:, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0]) + \
                   coor[:, 2] * (self.grid_size[1] * self.grid_size[0]) + \
                   coor[:, 1] *  self.grid_size[0] + coor[:, 0]
        coor_1d[np.where(kept==False)] = (B * self.grid_size[2] * self.grid_size[1] * self.grid_size[0]).astype(np.longlong)
        #for i in range(num_points):
        #    if kept[i]:
        #        coor_1d[i]  = coor[i, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0]) + \
        #                     coor[i, 2] * (self.grid_size[1] * self.grid_size[0]) + \
        #                     coor[i, 1] *  self.grid_size[0] + coor[:, 0]
        #    else:
        #        coor_1d[i] = B * self.grid_size[2] * self.grid_size[1] * self.grid_size[0]

        return bev_feat, np.ascontiguousarray(coor_1d.astype(np.longlong))


    def __call__(self, data, info_dict):
        coor = self.get_lidar_coor(info_dict)
        bev_feat, lidar_coor_1d = self.precompute_voxel_info(coor)

        ##  combine all 6 images into on
        temp = np.concatenate(data, 0)
        data=[]
        data.append(temp)

        # append bev_feat and lidar_coor_1d
        data.append(bev_feat)
        data.append(lidar_coor_1d)

        return data, info_dict


class GetBEVFormerGeometry():

    def __init__(self, bev_size):
        # how to configure these params?
        self.bev_h = bev_size[0]
        self.bev_w = bev_size[1]
        self.num_points_in_pillar = 4
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_w = self.pc_range[3] - self.pc_range[0]

        self.rotate_prev_bev =  True
        self.rotate_center = [100, 100]

    def get_reference_points(self, H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, dtype=np.float32):
        """
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            NP array: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = np.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype).reshape(-1, 1, 1)
            zs = np.broadcast_to(zs, (num_points_in_pillar, H, W)) / Z

            xs = np.linspace(0.5, W - 0.5, W, dtype=dtype).reshape(1, 1, W)
            xs = np.broadcast_to(xs, (num_points_in_pillar, H, W)) / W

            ys = np.linspace(0.5, H - 0.5, H, dtype=dtype).reshape(1, H, 1)
            ys = np.broadcast_to(ys, (num_points_in_pillar, H, W)) / H

            ref_3d = np.stack((xs, ys, zs), -1)

            ref_3d = np.transpose(ref_3d, (0, 3, 1, 2))
            B, C, H, W = ref_3d.shape
            ref_3d = np.transpose(ref_3d.reshape(B, C, H*W), (0, 2, 1))
            ref_3d = np.repeat(ref_3d[None], bs, 0)

            return ref_3d

    def point_sampling(self, reference_points, pc_range,  info_dict):

        lidar2img = []
        lidar2img.append(info_dict['lidar2imgs'])
        lidar2img = np.asarray(lidar2img)
        #lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.copy()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = np.concatenate(
            (reference_points, np.ones_like(reference_points[..., :1])), -1)

        reference_points = np.transpose(reference_points, (1, 0, 2, 3)) # 4x1x2500x4
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]

        reference_points = reference_points.reshape(D, B, 1, num_query, 4)
        reference_points = np.expand_dims(np.repeat(reference_points, num_cam, 2), -1)

        lidar2img = lidar2img.reshape(1, B, num_cam, 1, 4, 4)
        lidar2img = np.repeat(lidar2img, D, 0)
        lidar2img = np.repeat(lidar2img, num_query, 3)

        reference_points_cam = np.squeeze(np.matmul(lidar2img.astype(np.float32),
                                            reference_points.astype(np.float32)), -1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / np.maximum(
            reference_points_cam[..., 2:3], np.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= info_dict['pad_shape'][1]
        reference_points_cam[..., 1] /= info_dict['pad_shape'][0]

        # clip reference_points_cam for quantization
        reference_points_cam = np.clip(reference_points_cam, 0.0, 1.0)

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        bev_mask = np.nan_to_num(bev_mask)

        reference_points_cam = np.transpose(reference_points_cam, (2, 1, 3, 0, 4))
        bev_mask = np.squeeze(np.transpose(bev_mask, (2, 1, 3, 0, 4)), -1)

        return reference_points_cam, bev_mask


    def precompute_bev_info(self, info_dict):

        # Pre-compute the voxel info 
        ref_3d = self.get_reference_points(
            self.bev_h, self.bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar,
            dim='3d', bs=1, dtype=np.float32)

        # Get image coors corresponding to ref_3d. bev_mask indicates valid coors
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, info_dict)

        bev_valid_indices = []
        bev_valid_indices_count = []
        for mask_per_img in bev_mask:
            #index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            nzindex = np.squeeze(np.nonzero(np.sum(mask_per_img[0], -1))[0])
            index_query_per_img = np.ones(self.bev_h*self.bev_w) * self.bev_h*self.bev_w
            index_query_per_img[:len(nzindex)] = nzindex
            bev_valid_indices.append(index_query_per_img)
            bev_valid_indices_count.append(len(nzindex))

        # Get bev_mask_count from bev_mask for encoder spatial_cross_attention
        bev_mask_count = np.sum(bev_mask, -1) > 0
        bev_mask_count = np.sum(np.transpose(bev_mask_count, (1, 2, 0)), -1)
        bev_mask_count = np.clip(bev_mask_count, a_min=1.0, a_max=None).astype(np.float32)
        bev_mask_count = bev_mask_count[..., None]

        can_bus = np.expand_dims(info_dict['can_bus'], 0)

        delta_x = np.array([info_dict['can_bus'][0]])
        delta_y = np.array([info_dict['can_bus'][1]])
        ego_angle = np.array([info_dict['can_bus'][-2] / np.pi * 180])
        grid_length_y = self.real_h / self.bev_h
        grid_length_x = self.real_w / self.bev_w
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w

        return reference_points_cam, bev_mask_count, \
               np.expand_dims(np.concatenate(bev_valid_indices, axis=0), axis=1).astype(np.int32), \
               np.array(bev_valid_indices_count).astype(np.int32), \
               np.array([[shift_x[0],shift_y[0]]]).astype(np.float32), can_bus.astype(np.float32)


    # Based on torchvision.transforms.functional._get_inverse_affine_matrix()
    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear, inverted=True):
        # Helper method to compute inverse matrix for affine transformation

        # Pillow requires inverse affine transformation matrix:
        # Affine matrix is : M = T * C * RotateScaleShear * C^-1
        #
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RotateScaleShear is rotation with scale and shear matrix
        #
        #       RotateScaleShear(a, s, (sx, sy)) =
        #       = R(a) * S(s) * SHy(sy) * SHx(sx)
        #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
        #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
        #         [ 0                    , 0                                      , 1 ]
        # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
        # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
        #          [0, 1      ]              [-tan(s), 1]
        #
        # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1
        rot = math.radians(angle)
        sx = math.radians(shear[0])
        sy = math.radians(shear[1])

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = math.cos(rot - sy) / math.cos(sy)
        b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
        c = math.sin(rot - sy) / math.cos(sy)
        d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

        if inverted:
            # Inverted rotation matrix with scale and shear
            # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
            matrix = [d, -b, 0.0, -c, a, 0.0]
            matrix = [x / scale for x in matrix]
            # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
            matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
            matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
            # Apply center translation: C * RSS^-1 * C^-1 * T^-1
            matrix[2] += cx
            matrix[5] += cy
        else:
            matrix = [a, b, 0.0, c, d, 0.0]
            matrix = [x * scale for x in matrix]
            # Apply inverse of center translation: RSS * C^-1
            matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
            matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
            # Apply translation and center : T * C * RSS * C^-1
            matrix[2] += cx + tx
            matrix[5] += cy + ty

        return matrix


    # Based on torchvision.transforms._functional_tensor._gen_affine_grid()
    def gen_affine_grid(self, theta, w, h, ow, oh):
        # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
        # AffineGridGenerator.cpp#L18
        # Difference with AffineGridGenerator is that:
        # 1) we normalize grid values after applying theta
        # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

        d = 0.5
        base_grid = np.empty([1, oh, ow, 3], dtype=theta.dtype)
        x_grid = np.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, ow)
        base_grid[..., 0] = np.copy(x_grid)
        y_grid = np.expand_dims(np.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, oh), -1)
        base_grid[..., 1] = np.copy(y_grid)
        base_grid[..., 2].fill(1)

        rescaled_theta = np.transpose(theta, (0, 2, 1)) / np.array([0.5 * w, 0.5 * h], dtype=theta.dtype)
        output_grid = np.matmul(base_grid.reshape(1, oh * ow, 3), rescaled_theta)
        return output_grid.reshape(1, oh, ow, 2)


    def compute_rotation_matrix(self, info_dict):
        height = self.bev_h
        width  = self.bev_w
        oh = height
        ow = width
        dtype = np.float32

        center_f = [0.0, 0.0]
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(self.rotate_center, [width, height])]

        angle = info_dict['can_bus'][-1]
        matrix = self.get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

        theta = np.array(matrix, dtype=dtype).reshape(1, 2, 3)
        grid = self.gen_affine_grid(theta, width, height, ow, oh)
        return grid


    def __call__(self, data, info_dict):
        reference_points_cam, bev_mask_count, bev_valid_indices, bev_valid_indices_count, shift_yx, can_bus = \
            self.precompute_bev_info(info_dict)

        rotation_grid = self.compute_rotation_matrix(info_dict)

        ## for bevformer, combine all 6 images into on
        temp = np.concatenate(data, 0)
        data=[]
        data.append(temp)

        data.append(shift_yx)
        data.append(rotation_grid)
        data.append(reference_points_cam)
        data.append(bev_mask_count)
        data.append(bev_valid_indices)
        # Not needed for the latest model (bevformer_tiny_plus_480x800_20250408.onnx)
        #data.append(bev_valid_indices_count)
        data.append(can_bus)

        if info_dict['prev_bev_exist'] is False:
            data.append(np.zeros((self.bev_h*self.bev_w, 1, 256), dtype=np.float32))
        else:
            data.append(info_dict['prev_bev'])

        #info_dict['bev_h'] = self.bev_h
        #info_dict['bev_w'] = self.bev_w
        return data, info_dict

class GetFCOS3DGeometry():
    def __init__(self):
        pass

    def __call__(self, img, info_dict):
        cam2img = info_dict['intrins']
        pad_cam2img = np.eye(4, dtype=np.float32)
        pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
        inv_pad_cam2img = np.linalg.inv(pad_cam2img).transpose(1, 0)

        data = []
        data.append(img)
        data.append(pad_cam2img)
        data.append(inv_pad_cam2img)

        return data, info_dict


class GetFastBEVGeometry():
    """ Create input data including camera frustum (i.e. 3D volume around ego vehicle)
        which is needed to run FastBEV. This camera frustum is constructed based on 
        cameras' intrinsic/extrinsic params.
    """
    def __init__(self, crop):
        # how to configure these params?
        self.feats_size        = [6, 64, 64, 176]
        self.n_voxels          = [200, 200, 4]
        self.voxel_size        = [0.5, 0.5, 1.5]
        self.point_cloud_range = [-50, -50, -5, 50, 50, 3]

        #self.crop = crop
        #self.downsample = 16
        #self.grid_config = {
        #    'x': [-51.2, 51.2, 0.8],
        #    'y': [-51.2, 51.2, 0.8],
        #    'z': [-5, 3, 8],
        #    'depth': [1.0, 60.0, 1.0],
        #}
        #self.out_channels = 64

        #self.create_grid_infos(**self.grid_config)

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = np.eye(3)
        intrinsic[:2] /= stride
        for cam_id in range(len(img_meta['lidar2imgs'])):
            extrinsic = img_meta['lidar2imgs'][cam_id]
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])

        return np.stack(projection)

    @staticmethod
    def get_points(n_voxels, voxel_size, origin):
        points = np.stack(
            np.meshgrid(np.arange(n_voxels[0]),
                        np.arange(n_voxels[1]),
                        np.arange(n_voxels[2]), indexing='ij')
        )
        new_origin = origin - n_voxels / 2.0 * voxel_size
        points = points * voxel_size.reshape(3, 1, 1, 1) + new_origin.reshape(3, 1, 1, 1)
        return points

    @staticmethod
    def get_augmented_img_params(img_meta):
        fH, fW  = img_meta['pad_shape']
        H, W, _ = img_meta['data_shape']

        resize = float(fW)/float(W)
        resize_dims = (int(W * resize), int(H * resize))

        newW, newH = resize_dims
        crop_h_start = (newH - fH) // 2
        crop_w_start = (newW - fW) // 2
        crop = (crop_w_start, crop_h_start, crop_w_start + fW, crop_h_start + fH)

        return resize, resize_dims, crop

    @staticmethod
    def scale_augmented_img_params(post_rot, post_tran, resize_r, resize_dims, crop):
        post_rot *= resize_r
        post_tran -= np.array(crop[:2])

        A = get_rot(0)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = np.matmul(A, -b) + b
        post_rot = np.matmul(A, post_rot)
        post_tran = np.matmul(A, post_tran) + b

        ret_post_rot, ret_post_tran = np.eye(3), np.zeros(3)
        ret_post_rot[:2, :2] = post_rot
        ret_post_tran[:2] = post_tran

        return ret_post_rot, ret_post_tran

    def rts2proj(self, img_meta, post_rot=None, post_tran=None):
        if img_meta is None:
            return None

        for cam_id in range(len(img_meta['lidar2imgs'])):
            lidar2cam = img_meta['lidar2cams'][cam_id]
            intrinsic = img_meta['intrins'][0][cam_id]

            viewpad = np.eye(4)
            if post_rot is not None:
                assert post_tran is not None, [post_rot, post_tran]
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = post_rot @ intrinsic
                viewpad[:3, 2] += post_tran
            else:
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            img_meta['lidar2imgs'][cam_id] = (viewpad @ lidar2cam).astype(np.float32)

        return img_meta

    def precompute_volume_info(self, points, projection):
        """
        function: 2d feature + predefined point cloud -> 3d volume
        """
        n_images, n_channels, height, width = self.feats_size
        n_x, n_y, n_z = points.shape[-3:]

        points = np.broadcast_to(points.reshape(1, 3, -1), (n_images, 3, n_x*n_y*n_z))
        points = np.concatenate((points, np.ones_like(points[:, :1])), axis=1)
    
        # ego_to_cam
        points_2d_3 = np.matmul(projection, points)  # lidar2img
        x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().astype(np.longlong)  # [6, 160000]
        y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().astype(np.longlong)  # [6, 160000]
        z = points_2d_3[:, 2]  # [6, 160000]
        valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 160000]

        # xy coordinate
        xy_coor = y * width + x

        coor      = np.full((1, xy_coor.shape[1]), width*height*n_images)
        cum_valid = np.full((1, xy_coor.shape[1]), True)
        cum_valid = cum_valid[0]

        for i in reversed(range(n_images)):
            valid_idx = np.multiply(cum_valid, valid[i]).astype(bool)
            coor[0, valid_idx] = xy_coor[i, valid_idx] + i*width*height
            cum_valid[valid_idx] = False

        return coor[0]

    def precompute_proj_info(self, data, info_dict, prev_img_metas=None):
        xy_coor_list   = []

        n_times = 1
        if 'num_bev_temporal_frames' in info_dict:
            n_times = info_dict['num_bev_temporal_frames'] + 1
        stride_i = math.ceil(data[0].shape[-1] / self.feats_size[-1])

        for batch_id in range(len(info_dict['intrins'])):
            img_meta_list = []
            img_meta = copy.deepcopy(info_dict)

            if isinstance(img_meta["pad_shape"], list):
                img_meta["pad_shape"] = img_meta["pad_shape"][0]

            # add to img_meta_list
            img_meta_list.append(img_meta)

            # update adjacent img_metas:
            #  refer to get_adj_data_info() and refine_data_info()
            #  in projects/FastBEV/fast_bev/nuscenes_dataset.py
            for i in range(n_times - 1):
                prev_img_meta = copy.deepcopy(prev_img_metas[i])

                egocurr2global = img_meta['ego2globals'][batch_id][0]
                egoadj2global = prev_img_meta['ego2globals'][batch_id][0]
                lidar2ego = img_meta['lidar2ego']
                lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                    @ egoadj2global @ lidar2ego

                for cam_id in range(len(img_meta['lidar2imgs'])):
                    mat = lidaradj2lidarcurr @ img_meta['cam2lidars'][cam_id]
                    prev_img_meta['cam2lidars'][cam_id] = mat
                    prev_img_meta['lidar2cams'][cam_id] = np.linalg.inv(mat)

                    # obtain lidar to image transformation matrix
                    prev_img_meta['intrins'][batch_id][cam_id] = \
                        img_meta['intrins'][batch_id][cam_id]
                    intrin = prev_img_meta['intrins'][batch_id][cam_id]
                    viewpad = np.eye(4)
                    viewpad[:intrin.shape[0], :intrin.shape[1]] = intrin
                    prev_img_meta['lidar2imgs'][cam_id] = \
                        viewpad @ img_meta['lidar2cams'][cam_id]

                # Get augmented scaled params (lidar2imgs):
                resize_r, resize_dims, crop = self.get_augmented_img_params(prev_img_meta)
                post_rot, post_tran = self.scale_augmented_img_params(
                    np.eye(2), np.zeros(2),
                    resize_r=resize_r,
                    resize_dims=resize_dims,
                    crop=crop
                )

                prev_img_meta = self.rts2proj(prev_img_meta, post_rot, post_tran)

                if isinstance(prev_img_meta["pad_shape"], list):
                    prev_img_meta["pad_shape"] = prev_img_meta["pad_shape"][0]

                # add to img_meta_list
                img_meta_list.append(prev_img_meta)

            # precompute projection
            for seq_id in range(n_times):
                img_meta = img_meta_list[seq_id]

                projection = self._compute_projection(img_meta, stride_i, noise=0)

                # self.style in ['v1', 'v2']:
                n_voxels, voxel_size = self.n_voxels, self.voxel_size
                origin = (np.array(self.point_cloud_range[:3]) + 
                          np.array(self.point_cloud_range[3:])) / 2

                points = self.get_points(n_voxels=np.array(n_voxels),
                                         voxel_size=np.array(voxel_size),
                                         origin=origin)
                xy_coor = self.precompute_volume_info(points, projection).astype(np.int32)
                xy_coor_list.append(xy_coor)

        if n_times > 1:
            return np.stack(xy_coor_list)
        else:
            return xy_coor_list[0]


    def get_temporal_feats(self, info_dict):
        prev_feat = None
        prev_img_meta = info_dict
        prev_feats = []
        prev_img_metas = []

        if 'queue' in info_dict:
            num_prevs   = info_dict['num_bev_temporal_frames']
            queue_mem   = copy.deepcopy(info_dict['queue_mem'])
            feats_queue = info_dict['queue']
            del info_dict['queue_mem']
            del info_dict['queue']

            # Support only batch_size = 1s
            for i in range(1, num_prevs+1):
                cur_sample_idx = info_dict['sample_idx']

                if i > feats_queue.qsize() or \
                    info_dict['scene_token'] != queue_mem[cur_sample_idx - i]['img_meta']['scene_token']:
                    if prev_feat is None:
                        #prev_feats.append(np.zeros(self.feats_size, dtype=img.dtype))
                        prev_feats.append(np.zeros(self.feats_size, dtype=np.float32))
                        prev_img_metas.append(prev_img_meta)
                    else:
                        prev_feats.append(prev_feat)
                        prev_img_metas.append(prev_img_meta)
                else:
                    prev_feat = queue_mem[cur_sample_idx - i]['feature_map']
                    prev_img_meta = queue_mem[cur_sample_idx - i]['img_meta']
                    prev_feats.append(prev_feat)
                    prev_img_metas.append(prev_img_meta)

        return np.concatenate(prev_feats, axis=0), prev_img_metas

    def __call__(self, data, info_dict):

        # get previous temporal infos
        prev_feats_map = None
        prev_input_metas = None

        if 'num_bev_temporal_frames' in info_dict and info_dict['num_bev_temporal_frames'] > 0:
            prev_feats_map, prev_input_metas = self.get_temporal_feats(info_dict)
        xy_coors = self.precompute_proj_info(data, info_dict, prev_img_metas=prev_input_metas)

        ## for fastbev, combine all 6 images into one
        temp = np.concatenate(data, 0)
        data=[]
        data.append(temp)

        data.append(xy_coors)
        if 'num_bev_temporal_frames' in info_dict and info_dict['num_bev_temporal_frames'] > 0:
            data.append(prev_feats_map)

        return data, info_dict


class GetStreamPETRGeometry():
    def __init__(self):
        # Params needed to generate coords3d: How to make them configurable?
        # Batch size
        self.B              = 1
        self.C              = 256
        self.H              = 16
        self.W              = 44
        self.num_propagated = 128
        self.memory_len     = 512
        self.embed_dims     = 256
        self.prev_scene_token = None

        self.LID            = True
        self.stride         = 16
        self.depth_num      = 64
        self.depth_start    = 1
        self.pc_range       = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]).astype(np.float32)
        self.position_range = np.array([-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]).astype(np.float32)

        if self.LID:
            index  = np.arange(start=0, stop=self.depth_num, step=1).astype(np.float32)
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            self.coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = np.arange(start=0, stop=self.depth_num, step=1).astype(np.float32)
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            self.coords_d = self.depth_start + bin_size * index

        self.pseudo_reference_points = _pseudo_referernce_points_128


    def prepare_location(self, info_dict):
        pad_h, pad_w = info_dict['pad_shape']
        bs, n, h, w = self.B, len(info_dict['lidar2imgs']), self.H, self.W

        shifts_x = (np.arange(
            0, self.stride*w, step=self.stride).astype(np.float32) + self.stride // 2 ) / pad_w
        shifts_y = (np.arange(
            0, h * self.stride, step=self.stride).astype(np.float32) + self.stride // 2) / pad_h

        shift_y, shift_x = np.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        location = np.stack((shift_x, shift_y), axis=1)
        location = np.tile(location.reshape(h, w, 2)[None], (bs*n, 1, 1, 1))

        return location


    def create_coords3d(self, location, info_dict):
        eps = 1e-5
        BN, H, W, _ = location.shape

        img2lidar = []
        for i in range(len(info_dict['lidar2imgs'])):
            img2lidar.append(np.linalg.inv(info_dict['lidar2imgs'][i]))
        img2lidars = np.expand_dims(np.asarray(img2lidar), 0)
        intrinsics = info_dict['post_intrins']

        B = img2lidars.shape[0]
        intrinsic = np.stack([intrinsics[..., 0, 0], intrinsics[..., 1, 1]], axis=-1)
        intrinsic = np.abs(intrinsic) / 1e3
        intrinsic = np.tile(intrinsic, (1, H*W, 1)).reshape(B, -1, 2)
        LEN = intrinsic.shape[1]
        num_sample_tokens = LEN

        pad_h, pad_w  = info_dict['pad_shape']
        location[..., 0] = location[..., 0] * pad_w
        location[..., 1] = location[..., 1] * pad_h

        D = self.coords_d.shape[0]

        location = location.reshape(B, num_sample_tokens, 1, 2)
        topk_centers = np.tile(location, (1, 1, D, 1))
        coords_d = np.tile(self.coords_d.reshape(1, 1, D, 1), (B, num_sample_tokens, 1 , 1))
        coords = np.concatenate([topk_centers, coords_d], axis=-1)
        coords = np.concatenate((coords, np.ones_like(coords[..., :1])), axis=-1)
        coords[..., :2] = coords[..., :2] * np.maximum(coords[..., 2:3], np.ones_like(coords[..., 2:3])*eps)
        coords = np.expand_dims(coords, axis=-1)

        img2lidars = np.tile(img2lidars.reshape(BN, 1, 1, 4, 4), (1, H*W, D, 1, 1)).reshape(B, LEN, D, 4, 4)

        coords3d = np.squeeze(np.matmul(img2lidars, coords), -1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)

        # for spatial alignment in focal petr
        cone = np.concatenate([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], axis=-1)

        return coords3d, cone


    def init_memory(self, prev_exists):
        B = prev_exists.shape[0]
        memory_embedding = np.zeros((B, self.memory_len, self.embed_dims), dtype=np.float32)
        memory_reference_point = np.zeros((B, self.memory_len, 3), dtype=np.float32)
        memory_timestamp = np.zeros((B, self.memory_len, 1), dtype=np.float32)
        memory_egopose = np.zeros((B, self.memory_len, 4, 4), dtype=np.float32)
        memory_velo = np.zeros((B, self.memory_len, 2), dtype=np.float32)

        return memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_velo


    def pre_update_memory(self, info_dict):
        x = info_dict['prev_exists']
        B = x.shape[0]
        prev_memory = info_dict['prev_memory']
        if prev_memory is None:
            memory_embedding, memory_reference_point, memory_timestamp, \
                memory_egopose, memory_velo = self.init_memory(x)
        else:
            ego_pose_inv = np.linalg.inv(info_dict['ego2globals'][0][0] @ info_dict['lidar2ego'])
            ego_pose_inv = np.expand_dims(ego_pose_inv, 0).astype(np.float32)
            timestamp = np.asarray([info_dict['timestamp']*1e-6])

            memory_embedding       = info_dict['prev_memory'][0]
            memory_reference_point = info_dict['prev_memory'][1]
            memory_timestamp       = info_dict['prev_memory'][2]
            memory_egopose         = info_dict['prev_memory'][3]
            memory_velo            = info_dict['prev_memory'][4]

            memory_timestamp += np.expand_dims(np.expand_dims(timestamp, -1), -1)
            memory_egopose = np.expand_dims(ego_pose_inv, 1) @ memory_egopose
            memory_reference_point = transform_reference_points(memory_reference_point, ego_pose_inv, reverse=False)
            memory_timestamp = memory_refresh(memory_timestamp[:, :self.memory_len], x)
            memory_reference_point = memory_refresh(memory_reference_point[:, :self.memory_len], x)
            memory_embedding = memory_refresh(memory_embedding[:, :self.memory_len], x)
            memory_egopose = memory_refresh(memory_egopose[:, :self.memory_len], x)
            memory_velo = memory_refresh(memory_velo[:, :self.memory_len], x)

        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            memory_reference_point[:, :self.num_propagated]  = memory_reference_point[:, :self.num_propagated] + (1 - x).reshape(B, 1, 1) * pseudo_reference_points
            memory_egopose[:, :self.num_propagated]  = memory_egopose[:, :self.num_propagated] + (1 - x).reshape(B, 1, 1, 1) * np.eye(4)

        return memory_embedding, memory_reference_point, memory_timestamp.astype(np.float32), \
               memory_egopose, memory_velo

    def get_ego_pose_and_timestamp(self, info_dict):
        ego_pose = info_dict['ego2globals'][0][0] @ info_dict['lidar2ego']
        ego_pose = np.expand_dims(ego_pose, 0).astype(np.float32)
        timestamp = np.asarray([info_dict['timestamp']*1e-6])

        return ego_pose, timestamp

    def __call__(self, data, info_dict):

        location = self.prepare_location(info_dict)
        coords_3d, cone = self.create_coords3d(location, info_dict)

        if info_dict['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = info_dict['scene_token']
            info_dict['prev_exists'] = np.array([0], dtype=np.int32)
        else:
            info_dict['prev_exists'] = np.array([1], dtype=np.int32)

        memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_velo = self.pre_update_memory(info_dict)

        ego_pose, timestamp = self.get_ego_pose_and_timestamp(info_dict)

        # Model inputs
        data_ = []
        # combine all 6 images into one
        data_.append(np.concatenate(data, 0))
        data_.append(memory_embedding)
        data_.append(memory_reference_point)
        data_.append(memory_timestamp)
        data_.append(memory_egopose)
        data_.append(memory_velo)
        data_.append(coords_3d)
        data_.append(cone)
        data_.append(ego_pose)
        data_.append(timestamp)

        return data_, info_dict


class GetFar3DGeometry():
    def __init__(self):
        # Params needed to generate coords3d: How to make them configurable?
        # Batch size
        self.B              = 1
        self.N              = 6
        #self.C              = 256
        #self.H              = 16
        #self.W              = 44
        self.num_propagated = 256
        self.memory_len     = 1024
        self.embed_dims     = 256
        self.prev_scene_token = None

        #self.LID            = True
        #self.stride         = 16
        #self.depth_num      = 64
        #self.depth_start    = 1
        self.img_feat_sizes = [[80, 120], [40, 60], [20, 30], [10, 15]]
        self.pc_range       = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]).astype(np.float32)
        #self.position_range = np.array([-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]).astype(np.float32)

        self.pseudo_reference_points = _pseudo_referernce_points_256

    def init_memory(self, prev_exists):
        B = prev_exists.shape[0]
        memory_embedding = np.zeros((B, self.memory_len, self.embed_dims), dtype=np.float32)
        memory_reference_point = np.zeros((B, self.memory_len, 3), dtype=np.float32)
        memory_timestamp = np.zeros((B, self.memory_len, 1), dtype=np.float32)
        memory_egopose = np.zeros((B, self.memory_len, 4, 4), dtype=np.float32)
        memory_velo = np.zeros((B, self.memory_len, 2), dtype=np.float32)

        return memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_velo


    def pre_update_memory(self, info_dict):
        x = info_dict['prev_exists']
        B = x.shape[0]
        prev_memory = info_dict['prev_memory']
        if prev_memory is None:
            memory_embedding, memory_reference_point, memory_timestamp, \
                memory_egopose, memory_velo = self.init_memory(x)
        else:
            ego_pose_inv = np.linalg.inv(info_dict['ego2globals'][0][0] @ info_dict['lidar2ego'])
            ego_pose_inv = np.expand_dims(ego_pose_inv, 0).astype(np.float32)
            timestamp = np.asarray([info_dict['timestamp']*1e-6])

            memory_embedding       = info_dict['prev_memory'][0]
            memory_reference_point = info_dict['prev_memory'][1]
            memory_timestamp       = info_dict['prev_memory'][2]
            memory_egopose         = info_dict['prev_memory'][3]
            memory_velo            = info_dict['prev_memory'][4]

            memory_timestamp += np.expand_dims(np.expand_dims(timestamp, -1), -1)
            memory_egopose = np.expand_dims(ego_pose_inv, 1) @ memory_egopose
            memory_reference_point = transform_reference_points(memory_reference_point, ego_pose_inv, reverse=False)
            memory_timestamp = memory_refresh(memory_timestamp[:, :self.memory_len], x)
            memory_reference_point = memory_refresh(memory_reference_point[:, :self.memory_len], x)
            memory_embedding = memory_refresh(memory_embedding[:, :self.memory_len], x)
            memory_egopose = memory_refresh(memory_egopose[:, :self.memory_len], x)
            memory_velo = memory_refresh(memory_velo[:, :self.memory_len], x)

        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            memory_reference_point[:, :self.num_propagated]  = memory_reference_point[:, :self.num_propagated] + (1 - x).reshape(B, 1, 1) * pseudo_reference_points
            memory_egopose[:, :self.num_propagated]  = memory_egopose[:, :self.num_propagated] + (1 - x).reshape(B, 1, 1, 1) * np.eye(4)

        return memory_embedding, memory_reference_point, memory_timestamp.astype(np.float32), \
               memory_egopose, memory_velo

   
    def prepare_sensor_matrices(self, info_dict):

        lidar2imgs = []
        img2lidars = []
        intrinsics = []
        for i in range(len(info_dict['lidar2imgs'])):
            lidar2imgs.append(info_dict['lidar2imgs'][i])
            img2lidars.append(np.linalg.inv(info_dict['lidar2imgs'][i]))
            intrin = np.eye(4).astype(np.float32)
            intrin[:3, :3] = info_dict['post_intrins'][0, i]
            intrinsics.append(intrin)

        lidar2imgs = np.expand_dims(np.asarray(lidar2imgs), 0)
        img2lidars = np.expand_dims(np.asarray(img2lidars), 0)
        intrinsics =  np.expand_dims(np.asarray(intrinsics), 0) / 1e3
        extrinsics = np.expand_dims(np.asarray(info_dict['lidar2cams']), 0)[..., :3, :]

        return intrinsics, extrinsics, lidar2imgs, img2lidars

    def get_ego_pose_and_timestamp(self, info_dict):
        ego_pose = info_dict['ego2globals'][0][0] @ info_dict['lidar2ego']
        ego_pose = np.expand_dims(ego_pose, 0).astype(np.float32)
        timestamp = np.asarray([info_dict['timestamp']*1e-6])

        return ego_pose, timestamp

    def __call__(self, data, info_dict):
        intrinsics, extrinsics, lidar2imgs, img2lidars = \
            self.prepare_sensor_matrices(info_dict)

        if info_dict['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = info_dict['scene_token']
            info_dict['prev_exists'] = np.array([0], dtype=np.int32)
        else:
            info_dict['prev_exists'] = np.array([1], dtype=np.int32)

        memory_embedding, memory_reference_point, memory_timestamp, \
            memory_egopose, memory_velo = self.pre_update_memory(info_dict)

        ego_pose, timestamp = self.get_ego_pose_and_timestamp(info_dict)

        # Model inputs
        data_ = []
        # combine all 6 images into one
        data_.append(np.concatenate(data, 0))
        data_.append(memory_embedding)
        data_.append(memory_reference_point)
        data_.append(memory_timestamp)
        data_.append(memory_egopose)
        data_.append(memory_velo)
        data_.append(intrinsics)
        data_.append(extrinsics)
        data_.append(lidar2imgs)
        data_.append(img2lidars)
        data_.append(ego_pose)
        data_.append(timestamp)

        return data_, info_dict
